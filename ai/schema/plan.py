from utils.logs import logger
from pydantic import BaseModel
from typing import Optional


class TaskResult(BaseModel):
    """Result of taking a task, with result and is_success required to be filled"""

    code: str = ""
    result: str
    is_success: bool


class Task(BaseModel):
    task_id: str = ""
    dependent_task_ids: list[str] = []  # Tasks prerequisite to this Task
    instruction: str = ""
    task_type: str = ""
    code: str = ""
    result: str = ""
    expected_output: str = ""
    is_success: bool = False
    is_finished: bool = False
    assignee: str = ""

    def reset(self):
        self.code = ""
        self.result = ""
        self.is_success = False
        self.is_finished = False

    def update_task_result(self, task_result: TaskResult):
        self.code = self.code + "\n" + task_result.code
        self.result = self.result + "\n" + task_result.result
        self.is_success = task_result.is_success


class Plan(BaseModel):
    """Plan is a sequence of tasks towards a goal."""

    goal: str = ""
    context: str = ""
    tasks: list[Task] = []
    task_map: dict[str, Task] = {}
    current_task_id: str = ""

    def _topological_sort(self, tasks: list[Task]):
        task_map = {task.task_id: task for task in tasks}
        dependencies = {task.task_id: set(task.dependent_task_ids) for task in tasks}
        sorted_tasks = []
        visited = set()

        def visit(task_id):
            if task_id in visited:
                return
            visited.add(task_id)
            for dependent_id in dependencies.get(task_id, []):
                visit(dependent_id)
            sorted_tasks.append(task_map[task_id])

        for task in tasks:
            visit(task.task_id)

        return sorted_tasks

    def add_tasks(self, tasks: list[Task]):
        """
        Integrates new tasks into the existing plan, ensuring dependency order is maintained.

        This method performs two primary functions based on the current state of the task list:
        1. If there are no existing tasks, it topologically sorts the provided tasks to ensure
        correct execution order based on dependencies, and sets these as the current tasks.
        2. If there are existing tasks, it merges the new tasks with the existing ones. It maintains
        any common prefix of tasks (based on task_id and instruction) and appends the remainder
        of the new tasks. The current task is updated to the first unfinished task in this merged list.

        Args:
            tasks (list[Task]): A list of tasks (may be unordered) to add to the plan.

        Returns:
            None: The method updates the internal state of the plan but does not return anything.
        """
        if not tasks:
            return

        # Topologically sort the new tasks to ensure correct dependency order
        new_tasks = self._topological_sort(tasks)

        if not self.tasks:
            # If there are no existing tasks, set the new tasks as the current tasks
            self.tasks = new_tasks

        else:
            # Find the length of the common prefix between existing and new tasks
            prefix_length = 0
            for old_task, new_task in zip(self.tasks, new_tasks):
                if (
                    old_task.task_id != new_task.task_id
                    or old_task.instruction != new_task.instruction
                ):
                    break
                prefix_length += 1

            # Combine the common prefix with the remainder of the new tasks
            final_tasks = self.tasks[:prefix_length] + new_tasks[prefix_length:]
            self.tasks = final_tasks

        # Update current_task_id to the first unfinished task in the merged list
        self._update_current_task()

        # Update the task map for quick access to tasks by ID
        self.task_map = {task.task_id: task for task in self.tasks}

    def reset_task(self, task_id: str):
        """
        Reset a task based on task_id, i.e. set Task.is_finished=False and request redo. This also resets all tasks depending on it.

        Args:
            task_id (str): The ID of the task to be reset.
        """
        if task_id in self.task_map:
            task = self.task_map[task_id]
            task.reset()
            # reset all downstream tasks that are dependent on the reset task
            for dep_task in self.tasks:
                if task_id in dep_task.dependent_task_ids:
                    # FIXME: if LLM generates cyclic tasks, this will result in infinite recursion
                    self.reset_task(dep_task.task_id)

        self._update_current_task()

    def _replace_task(self, new_task: Task):
        """
        Replace an existing task with the new input task based on task_id, and reset all tasks depending on it.

        Args:
            new_task (Task): The new task that will replace an existing one.

        Returns:
            None
        """
        assert new_task.task_id in self.task_map
        # Replace the task in the task map and the task list
        self.task_map[new_task.task_id] = new_task
        for i, task in enumerate(self.tasks):
            if task.task_id == new_task.task_id:
                self.tasks[i] = new_task
                break

        # Reset dependent tasks
        for task in self.tasks:
            if new_task.task_id in task.dependent_task_ids:
                self.reset_task(task.task_id)

        self._update_current_task()

    def _append_task(self, new_task: Task):
        """
        Append a new task to the end of existing task sequences

        Args:
            new_task (Task): The new task to be appended to the existing task sequence

        Returns:
            None
        """
        # assert not self.has_task_id(new_task.task_id), "Task already in current plan, use replace_task instead"
        if self.has_task_id(new_task.task_id):
            logger.warning(
                "Task already in current plan, should use replace_task instead. Overwriting the existing task."
            )

        assert all(
            [self.has_task_id(dep_id) for dep_id in new_task.dependent_task_ids]
        ), "New task has unknown dependencies"

        # Existing tasks do not depend on the new task, it's fine to put it to the end of the sorted task sequence
        self.tasks.append(new_task)
        self.task_map[new_task.task_id] = new_task
        self._update_current_task()

    def has_task_id(self, task_id: str) -> bool:
        return task_id in self.task_map

    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Lấy task theo task_id

        Args:
            task_id (str): ID của task cần lấy

        Returns:
            Task: Task object nếu tìm thấy, None nếu không tìm thấy
        """
        return self.task_map.get(task_id, None)

    def _update_current_task(self):
        self.tasks = self._topological_sort(self.tasks)
        # Update the task map for quick access to tasks by ID
        self.task_map = {task.task_id: task for task in self.tasks}

        current_task_id = ""
        for task in self.tasks:
            if not task.is_finished:
                current_task_id = task.task_id
                break
        self.current_task_id = current_task_id
        # TaskReporter().report({"tasks": [i.model_dump() for i in self.tasks], "current_task_id": current_task_id})

    @property
    def current_task(self) -> Optional[Task]:
        """Find current task to execute

        Returns:
            Task: the current task to be executed
        """
        return self.task_map.get(self.current_task_id, None)

    def finish_current_task(self):
        """Finish current task, set Task.is_finished=True, set current task to next task"""
        if self.current_task_id and self.current_task:
            self.current_task.is_finished = True
            self._update_current_task()  # set to next task

    def finish_all_tasks(self):
        "Finish all tasks."
        while self.current_task:
            self.finish_current_task()

    def is_plan_finished(self) -> bool:
        """Check if all tasks are finished"""
        return all(task.is_finished for task in self.tasks)

    def get_finished_tasks(self) -> list[Task]:
        """return all finished tasks in correct linearized order

        Returns:
            list[Task]: list of finished tasks
        """
        return [task for task in self.tasks if task.is_finished]

    def append_task(
        self,
        task_id: str,
        dependent_task_ids: list[str],
        instruction: str,
        assignee: str,
        task_type: str = "",
        expected_output: str = "",
    ):
        """
        Append a new task with task_id (number) to the end of existing task sequences.
        If dependent_task_ids is not empty, the task will depend on the tasks with the ids in the list.
        Note that the assignee should be the 'name' of the role.
        """
        new_task = Task(
            task_id=task_id,
            dependent_task_ids=dependent_task_ids,
            instruction=instruction,
            assignee=assignee,
            task_type=task_type,
            expected_output=expected_output,
        )
        return self._append_task(new_task)

    def replace_task(
        self,
        task_id: str,
        new_dependent_task_ids: list[str],
        new_instruction: str,
        new_assignee: str,
        task_type: str,
        expected_output: str,
    ):
        """Replace an existing task (can be current task) based on task_id, and reset all tasks depending on it."""
        new_task = Task(
            task_id=task_id,
            dependent_task_ids=new_dependent_task_ids,
            instruction=new_instruction,
            assignee=new_assignee,
            task_type=task_type,
            expected_output=expected_output,
        )
        return self._replace_task(new_task)
