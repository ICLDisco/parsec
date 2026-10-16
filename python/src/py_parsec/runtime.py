# PaRSEC runtime management - Python wrapper

class ParsecRuntime:
    """PaRSEC runtime wrapper"""
    
    def __init__(self, context=None):
        self._context = context
        print("Created PaRSEC runtime")
    
    def start(self):
        """Start the runtime"""
        if self._context is not None:
            self._context.start()
        print("Started PaRSEC runtime")
    
    def wait(self):
        """Wait for runtime completion"""
        if self._context is not None:
            self._context.wait()
        print("PaRSEC runtime completed")
    
    def test(self):
        """Test if runtime is complete"""
        if self._context is not None:
            return self._context.test()
        return 1

class ParsecScheduler:
    """PaRSEC scheduler wrapper"""
    
    def __init__(self, context=None):
        self._context = context
        self._started = False
        print("Created PaRSEC scheduler")
    
    def start(self):
        """Start the scheduler"""
        if self._context is not None and not self._started:
            self._context.start()
            self._started = True
        print("Started PaRSEC scheduler")
    
    def stop(self):
        """Stop the scheduler"""
        if self._context is not None and self._started:
            self._context.wait()
            self._started = False
        print("Stopped PaRSEC scheduler")
    
    @property
    def started(self):
        """Check if scheduler is started"""
        return self._started
