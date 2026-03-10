import sys
from pathlib import Path


# Zamenjuje stdout sa stoud + upis u fajl
# koristi se kao context manager
class Logger:
    def __init__(self, path):
        self.terminal = sys.stdout

        self.log_path = Path(path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.log_path.exists()
        self.log = open(self.log_path, 'a', buffering=1, encoding='utf-8')

        if file_exists:
            self.log.write(f"\n{'=' * 50}\n")
            self.log.write(f"New session started\n")
            self.log.write(f"{'=' * 50}\n\n")

        self.prev_message = None

    def write(self, message):
        self.terminal.write(message)

        # tqdm update, skip it and save it
        if '\r' in message and not message.endswith('\n'):
            self.prev_message = message.replace('\r', '')
            return
        # tqdm line finished
        if self.prev_message is not None and message == '\n':
            self.log.write(self.prev_message + '\n')
            self.prev_message = None
            return

        self.log.write(message)

    @property
    def encoding(self):
        return self.terminal.encoding

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        if not self.log.closed:
            self.log.close()

    def __enter__(self):
        # Save the original stdout and redirect to logger
        self.old_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original stdout and close file
        self.close()
        sys.stdout = self.old_stdout
        print(f"\nLogs saved to {self.log_path}")
        return False
