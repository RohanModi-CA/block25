import pickle
import msgpack
from dataclasses import asdict
from tracking_classes import Track1Params, DetectionRecord, FrameDetections, VideoCentroids

# --- THE HACK: Trick pickle into finding the classes in the main namespace ---
import __main__
__main__.Track1Params = Track1Params
__main__.DetectionRecord = DetectionRecord
__main__.FrameDetections = FrameDetections
__main__.VideoCentroids = VideoCentroids

def rescue_data():
    old_pkl_file = 'tracking_results_track1.pkl'
    new_msgpack_file = 'tracking_results_track1.msgpack'

    print(f"Loading brittle pickle: {old_pkl_file}...")
    try:
        with open(old_pkl_file, 'rb') as f:
            vc = pickle.load(f)
    except FileNotFoundError:
        print("Pickle file not found. If you don't have one, you can skip this rescue script.")
        return

    print("Successfully rescued data into memory!")
    
    # Convert safely to MessagePack
    print(f"Writing to robust MessagePack: {new_msgpack_file}...")
    with open(new_msgpack_file, 'wb') as f:
        # asdict(vc) turns the whole object tree into a standard dictionary!
        f.write(msgpack.packb(asdict(vc)))
        
    print("Migration complete. You can now safely delete the .pkl file and use track2.")

if __name__ == '__main__':
    rescue_data()
