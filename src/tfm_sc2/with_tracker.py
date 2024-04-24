from codecarbon import OfflineEmissionsTracker


class WithTracker:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tracker = OfflineEmissionsTracker(country_iso_code="ESP", measure_power_secs=10)
        self._emissions_per_task = dict()
        self._current_tracker_task = None

    def start_task(self, task_name: str):
        if self._current_tracker_task is not None and self._current_tracker_task != task_name:
            self.logger.warning(f"Auto-closing previous task {self._current_tracker_task}. This should not happen, please review your code.")
            self.stop_task()
        self._current_tracker_task = task_name
        self._tracker.start_task(task_name)

    def stop_task(self):
        if self._current_tracker_task is None:
            self.logger.warning("Called stop_tracking without any task being tracker. There's probably some error in your tracking logic")
            return

        task_emissions = self._tracker.stop_task()
        self._emissions_per_task[self._current_tracker_task] += task_emissions
        self._current_tracker_task = None

    def stop_tracker(self):
        self._tracker.stop()