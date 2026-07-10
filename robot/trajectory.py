import numpy as np


def _normalize(vec):
    arr = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return arr / norm


class EndEffectorPath:
    def __init__(self, waypoints=None, segment_durations=None):
        self.waypoints = []
        self.segment_durations = np.array([], dtype=float)

        if waypoints is not None:
            for waypoint in waypoints:
                self.add_waypoint(waypoint)

        if segment_durations is None:
            if len(self.waypoints) >= 2:
                self.segment_durations = np.ones(len(self.waypoints) - 1, dtype=float)
        else:
            self.set_segment_durations(segment_durations)

    @classmethod
    def from_waypoints(cls, waypoints, segment_durations=None, min_duration=0.1, linear_speed=1.0, angular_speed=1.0):
        traj = cls(waypoints=waypoints)
        if segment_durations is None:
            traj.set_auto_durations(
                linear_speed=linear_speed,
                angular_speed=angular_speed,
                min_duration=min_duration,
            )
        else:
            traj.set_segment_durations(segment_durations)
        return traj

    def add_waypoint(self, waypoint):
        pos, direction = waypoint
        pos = np.asarray(pos, dtype=float).copy()
        direction = _normalize(direction)
        if pos.shape != (3,) or direction.shape != (3,):
            raise ValueError("末端路径 waypoint 必须是 (pos[3], dir[3])")
        self.waypoints.append((pos, direction))

    def set_segment_durations(self, segment_durations):
        durations = np.asarray(segment_durations, dtype=float)
        expected = max(len(self.waypoints) - 1, 0)
        if len(durations) != expected:
            raise ValueError(f"segment_durations 长度应为 {expected}")
        if np.any(durations <= 0):
            raise ValueError("segment_durations 必须大于 0")
        self.segment_durations = durations

    def set_auto_durations(self, linear_speed=1.0, angular_speed=1.0, min_duration=0.1):
        if len(self.waypoints) < 2:
            self.segment_durations = np.array([], dtype=float)
            return

        linear_speed = max(float(linear_speed), 1e-6)
        angular_speed = max(float(angular_speed), 1e-6)
        durations = []
        for (p0, d0), (p1, d1) in zip(self.waypoints[:-1], self.waypoints[1:]):
            linear_dt = float(np.linalg.norm(p1 - p0) / linear_speed)
            cosine = float(np.clip(np.dot(d0, d1), -1.0, 1.0))
            angular_dt = float(np.arccos(cosine) / angular_speed)
            duration = max(linear_dt, angular_dt)
            durations.append(max(duration, min_duration))
        self.segment_durations = np.asarray(durations, dtype=float)

    @property
    def total_duration(self):
        return float(np.sum(self.segment_durations))

    def is_empty(self):
        return len(self.waypoints) == 0

    def waypoint_count(self):
        return len(self.waypoints)

    def get_waypoint(self, index):
        if self.is_empty():
            raise ValueError("trajectory 为空")
        index = int(np.clip(index, 0, len(self.waypoints) - 1))
        pos, direction = self.waypoints[index]
        return pos.copy(), direction.copy()

    def sample(self, t):
        if self.is_empty():
            raise ValueError("trajectory 为空")
        if len(self.waypoints) == 1 or self.total_duration <= 0:
            pos, direction = self.waypoints[0]
            return pos.copy(), direction.copy()

        time_s = float(np.clip(t, 0.0, self.total_duration))
        elapsed = 0.0
        for idx, duration in enumerate(self.segment_durations):
            next_elapsed = elapsed + duration
            if time_s <= next_elapsed or idx == len(self.segment_durations) - 1:
                alpha = 0.0 if duration <= 0 else (time_s - elapsed) / duration
                p0, d0 = self.waypoints[idx]
                p1, d1 = self.waypoints[idx + 1]
                pos = (1.0 - alpha) * p0 + alpha * p1
                direction = _normalize((1.0 - alpha) * d0 + alpha * d1)
                return pos, direction
            elapsed = next_elapsed
        pos, direction = self.waypoints[-1]
        return pos.copy(), direction.copy()

    def sample_progress(self, alpha):
        return self.sample(float(np.clip(alpha, 0.0, 1.0)) * self.total_duration)


class TrajectoryFollower:
    def __init__(self, trajectory=None):
        self.trajectory = None
        self.target_index = 0
        if trajectory is not None:
            self.set_trajectory(trajectory)

    def set_trajectory(self, trajectory):
        self.trajectory = trajectory
        if self.trajectory is None or self.trajectory.is_empty():
            self.target_index = 0
        else:
            # 第 0 个点当前末端位姿，从下一个 waypoint 开始跟踪。
            self.target_index = 1 if self.trajectory.waypoint_count() > 1 else 0

    def clear(self):
        self.trajectory = None
        self.target_index = 0

    def has_trajectory(self):
        return self.trajectory is not None and not self.trajectory.is_empty()

    def is_finished(self):
        return (not self.has_trajectory()) or self.target_index >= self.trajectory.waypoint_count() - 1

    def _target_reached(self, current_pos, current_dir, pos_tol, dir_tol):
        target_pos, target_dir = self.peek()
        pos_err = np.linalg.norm(np.asarray(current_pos, dtype=float) - target_pos)
        dir_err = np.linalg.norm(_normalize(current_dir) - target_dir)
        return pos_err <= pos_tol and dir_err <= dir_tol

    def update_target(self, current_pos, current_dir, pos_tol=0.1, dir_tol=0.1):
        if not self.has_trajectory():
            raise ValueError("trajectory 未设置")

        while self.target_index < self.trajectory.waypoint_count() - 1:
            if not self._target_reached(current_pos, current_dir, pos_tol, dir_tol):
                break
            self.target_index += 1
        return self.peek()

    def advance(self, dt):
        return self.peek()

    def peek(self):
        if not self.has_trajectory():
            raise ValueError("trajectory 未设置")
        return self.trajectory.get_waypoint(self.target_index)
