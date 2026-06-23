from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

class Tracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,           
            n_init=3,             
            max_iou_distance=0.7
        )
        self.stable_ids = set()   
        self.id_history = {}      

    def update(self, detections, frame):
        """
        detections: [{"bbox": [x1,y1,x2,y2], "conf": 0.9, "class": 2}, ...]
        Döndürür: track listesi ve kararlılık durumu
        """
        if not detections:
            return [], False

        # DeepSORT formatına çevir: [[x1,y1,w,h], conf, class]
        det_list = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            w = x2 - x1
            h = y2 - y1
            det_list.append(([x1, y1, w, h], d["conf"], d["class"]))

        tracks = self.tracker.update_tracks(det_list, frame=frame)

        active_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()

            
            self.id_history[track_id] = self.id_history.get(track_id, 0) + 1

            
            if self.id_history[track_id] >= 5:
                self.stable_ids.add(track_id)

            active_tracks.append({
                "id":     track_id,
                "bbox":   [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])],
                "stable": track_id in self.stable_ids
            })

        
        is_stable = any(t["stable"] for t in active_tracks) if active_tracks else False

        return active_tracks, is_stable

    def is_stable(self, track_id):
        return track_id in self.stable_ids