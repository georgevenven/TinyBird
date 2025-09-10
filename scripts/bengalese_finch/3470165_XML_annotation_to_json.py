import xml.etree.ElementTree as ET
import argparse 
import os
import json
   
"""
Input dir contains multiple folders of birds 0-10, each folder contains a XML file (Annotation.xml) and Wave folder (contains song)

XML structure
Sequences
    Sequence : position / length / num notes (persumably in samples), audio samples are at 32,000 hz 
        Notes: position, length, label (int)
"""

SAMPLING_RATE = 32_000
label_dict = {} # gets converted to json at the end 

def build_label_mapping(src_dir):
    """Scan all XMLs to map original labels (ints or letters) to integer IDs.
    Numeric labels keep their value. Non-numeric labels get IDs after max numeric."""
    labels = set()
    for folder in os.listdir(src_dir):
        folder_path = os.path.join(src_dir, folder)
        if os.path.isdir(folder_path):
            xml_path = os.path.join(folder_path, "Annotation.xml")
            if os.path.exists(xml_path):
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for seq in root.findall("Sequence"):
                    for note in seq.findall("Note"):
                        s = (note.find("Label").text or "").strip()
                        if s != "":
                            labels.add(s)
    # split numeric vs non-numeric
    max_num = -1
    label_map = {}
    non_numeric = []
    for s in labels:
        try:
            v = int(s)
            label_map[s] = v
            if v > max_num:
                max_num = v
        except ValueError:
            non_numeric.append(s)
    next_id = max_num + 1
    for s in sorted(non_numeric):
        label_map[s] = next_id
        next_id += 1
    return label_map

def main(args):
    global label_dict
    label_map = build_label_mapping(args["src_dir"])
    # top-level metadata
    label_dict["metadata"] = {"units": "ms"}
    rec_index = {}  # {(bird_id, wav): {"recording": {...}, "detected_events": [...]}}

    for folder in os.listdir(args["src_dir"]):
        folder_path = os.path.join(args["src_dir"], folder)

        bird_id = folder
        if os.path.isdir(folder_path):
            xml_path = os.path.join(folder_path,"Annotation.xml")
            if os.path.exists(xml_path):
                tree = ET.parse(xml_path)
                root = tree.getroot()

                for seq in root.findall("Sequence"):
                    wav = seq.find("WaveFileName").text
                    pos = int(seq.find("Position").text)
                    length = int(seq.find("Length").text)
                    nnotes = int(seq.find("NumNote").text)

                    notes = []
                    for note in seq.findall("Note"):
                        npos = int(note.find("Position").text)
                        nlen = int(note.find("Length").text)
                        raw = (note.find("Label").text or "").strip()
                        unit = label_map[raw]
                        notes.append((npos, nlen, unit))

                    # event = one XML <Sequence>, with absolute onsets/offsets
                    event_onset_ms = (1000.0 * pos) / SAMPLING_RATE
                    event_offset_ms = (1000.0 * (pos + length)) / SAMPLING_RATE
                    units = []
                    for npos, nlen, uid in notes:
                        u_on = pos + npos
                        u_off = u_on + nlen
                        units.append({
                            "onset_ms": (1000.0 * u_on) / SAMPLING_RATE,
                            "offset_ms": (1000.0 * u_off) / SAMPLING_RATE,
                            "id": uid
                        })

                    key = (bird_id, wav)
                    if key not in rec_index:
                        rec_index[key] = {
                            "recording": {
                                "sampling_rate_hz": SAMPLING_RATE,
                                "detected_vocalizations": 0,
                                "filename": wav,
                                "bird_id": bird_id
                            },
                            "detected_events": []
                        }
                    rec_index[key]["detected_events"].append({
                        "onset_ms": event_onset_ms,
                        "offset_ms": event_offset_ms,
                        "units": units
                    })
                    rec_index[key]["recording"]["detected_vocalizations"] += len(notes)

    # finalize and write JSON
    recordings = list(rec_index.values())
    label_dict["recordings"] = recordings
    # ensure dst_dir is a directory and write JSON inside it
    os.makedirs(args["dst_dir"], exist_ok=True)
    dst_path = os.path.join(args["dst_dir"], "annotations.json")
    with open(dst_path, "w") as f:
        json.dump(label_dict, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="format conversion args")

    parser.add_argument("--src_dir", type=str, help="full path, dir of xml file")
    parser.add_argument("--dst_dir", type=str, help="json output path")

    args = parser.parse_args()
    main(vars(args))