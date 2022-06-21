from FeatureCloud.app.engine.app import AppState, app_state, LogLevel, Role


@app_state('initial', Role.BOTH)
class InitialState(AppState):

    def register(self):
        self.register_transition('local_training', Role.BOTH)

    def run(self):
        return 'local_training'


@app_state('local_training', Role.BOTH)
class LocalTrainingState(AppState):
    def register(self):
        self.register_transition('global_aggregation', Role.COORDINATOR)
        self.register_transition('local_training', Role.PARTICIPANT)
        self.register_transition('terminal', Role.BOTH)

    def run(self):
        if self.is_coordinator:
            if True:#some condition
                return'global_aggregation'
            else:
                return 'terminal'
        else:
            if True:#some condition
                return 'local_training'
            else:
                return 'terminal'


@app_state('global_aggregation', Role.COORDINATOR)
class GlobalAggregateState(AppState):
    def register(self):
        self.register_transition('local_training', Role.COORDINATOR)

    def run(self):
        return 'local_training'

