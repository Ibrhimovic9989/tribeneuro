"""End-to-end TRIBE v2 brain prediction test."""
import warnings; warnings.filterwarnings('ignore')
import logging, os
import numpy as np
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tribev2 import TribeModel

model = TribeModel.from_pretrained(
    "facebook/tribev2",
    cache_folder="./cache",
    device="cpu",
    config_update={
        "data.text_feature.device": "cpu",
        "data.audio_feature.device": "cpu",
        "data.num_workers": 0,
    },
)
print("TRIBE v2 loaded\n")

events = model.get_events_dataframe(text_path="./test_data/test_stimulus.txt")
print(f"Events: {len(events)} rows, types: {events['type'].unique().tolist()}")
if 'Word' in events['type'].values:
    words = events[events['type']=='Word']['text'].tolist()
    print(f"Words: {' '.join(words)}")

preds, segments = model.predict(events, verbose=True)

print(f"\n{'='*50}")
print(f"BRAIN PREDICTION COMPLETE")
print(f"{'='*50}")
print(f"Shape: {preds.shape}")
print(f"  {preds.shape[0]} timesteps x {preds.shape[1]:,} vertices")
print(f"Range: [{preds.min():.4f}, {preds.max():.4f}]")
print(f"Mean |activation|: {np.mean(np.abs(preds)):.4f}")

mean_act = np.mean(np.abs(preds), axis=0)
top_10 = np.argsort(mean_act)[-10:][::-1]
print(f"\nTop 10 most active vertices:")
for i, v in enumerate(top_10):
    hemi = "Left" if v < 10242 else "Right"
    print(f"  {i+1}. Vertex {v} ({hemi} hemisphere): {mean_act[v]:.4f}")

os.makedirs('./neurodiverse_report', exist_ok=True)
np.save('./neurodiverse_report/text_brain_prediction.npy', preds)
print(f"\nSaved to ./neurodiverse_report/")
print("\n=== TRIBE v2 END-TO-END PIPELINE VERIFIED ===")
