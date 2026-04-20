# PaRSEC task management - Python wrapper

class TaskGraph:
    """Task graph for managing dependencies"""
    
    def __init__(self, context=None):
        self._context = context
        self._tasks = []
        self._dependencies = []
        print("Created PaRSEC task graph")
    
    def add_task(self, task):
        """Add a task to the graph"""
        self._tasks.append(task)
        print(f"Added task to graph: {task}")
    
    def add_dependency(self, from_task, to_task):
        """Add a dependency between tasks"""
        self._dependencies.append((from_task, to_task))
        print(f"Added dependency: {from_task} -> {to_task}")
    
    def submit_all(self):
        """Submit all tasks in the graph"""
        print(f"Submitting {len(self._tasks)} tasks")
        for task in self._tasks:
            if hasattr(task, 'submit'):
                task.submit()

class Task:
    """High-level task wrapper"""
    
    def __init__(self, context=None, function=None, inputs=None, outputs=None):
        self._context = context
        self._function = function
        self._inputs = inputs or []
        self._outputs = outputs or []
        print(f"Created task: {function}")
    
    def execute(self):
        """Execute the task function"""
        if self._function is not None:
            return self._function(*self._inputs)
        return None
    
    def submit(self):
        """Submit the task for execution"""
        print(f"Submitting task: {self._function}")
        return self.execute()

class DataDescriptor:
    """Data descriptor for task inputs/outputs"""
    
    def __init__(self, context=None, name="", shape=None, dtype=float):
        self._context = context
        self._name = name
        self._shape = shape or (1,)
        self._dtype = dtype
        # Create a simple data object
        self._data = None
        print(f"Created data descriptor: {name}, shape={shape}, dtype={dtype}")
    
    @property
    def name(self):
        return self._name
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def dtype(self):
        return self._dtype
    
    @property
    def data(self):
        return self._data
