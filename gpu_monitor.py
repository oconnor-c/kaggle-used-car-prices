import csv
import subprocess
import threading
import time
from datetime import datetime


class GPUMonitor:
    def __init__(self, interval=1.0, output_path='gpu_usage.png', gpu_index=0):
        self.interval = interval
        self.output_path = output_path
        self.gpu_index = gpu_index
        self._stop_event = threading.Event()
        self._thread = None
        self.samples = []

    def _read_gpu(self):
        command = [
            'nvidia-smi',
            f'--id={self.gpu_index}',
            '--query-gpu=utilization.gpu,memory.used,power.draw',
            '--format=csv,noheader,nounits',
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        row = next(csv.reader([result.stdout.strip()]))
        return {
            'time': time.time(),
            'gpu_utilization': float(row[0].strip()),
            'memory_used': float(row[1].strip()),
            'power_draw': float(row[2].strip()),
        }

    def _collect(self):
        while not self._stop_event.is_set():
            try:
                self.samples.append(self._read_gpu())
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                print(f'GPU 监控采样失败: {error}')
            self._stop_event.wait(self.interval)

    def start(self):
        self.samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._collect, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        if self.samples:
            self._save_plot()
            print(f'GPU 利用率图已保存到: {self.output_path}')

    def _save_plot(self):
        import matplotlib.pyplot as plt

        start_time = self.samples[0]['time']
        elapsed = [sample['time'] - start_time for sample in self.samples]
        utilization = [sample['gpu_utilization'] for sample in self.samples]
        memory_used = [sample['memory_used'] for sample in self.samples]
        power_draw = [sample['power_draw'] for sample in self.samples]

        figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        figure.suptitle(
            f'GPU monitoring ({datetime.fromtimestamp(start_time):%Y-%m-%d %H:%M:%S})'
        )
        axes[0].plot(elapsed, utilization)
        axes[0].set_ylabel('GPU utilization (%)')
        axes[0].set_ylim(0, 100)
        axes[1].plot(elapsed, memory_used)
        axes[1].set_ylabel('Memory used (MiB)')
        axes[2].plot(elapsed, power_draw)
        axes[2].set_ylabel('Power draw (W)')
        axes[2].set_xlabel('Elapsed time (s)')
        for axis in axes:
            axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(self.output_path, dpi=150)
        plt.close(figure)
