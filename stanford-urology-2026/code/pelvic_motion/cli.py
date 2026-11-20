"""CSV landmark track to CSV kinematics, with explicit calibration."""
import argparse,csv,json,pathlib
from .kinematics import calculate_kinematics,summarize_track

def main():
    parser=argparse.ArgumentParser(description="Analyze a supplied landmark track; not a clinical device")
    parser.add_argument("input_csv",help="CSV columns: time_s,x_px,y_px")
    parser.add_argument("output_csv")
    parser.add_argument("--mm-per-pixel",type=float,default=None)
    args=parser.parse_args()
    with open(args.input_csv,newline="",encoding="utf-8") as f: data=list(csv.DictReader(f))
    rows=calculate_kinematics([float(r["time_s"]) for r in data],
                              [float(r["x_px"]) for r in data],
                              [float(r["y_px"]) for r in data],args.mm_per_pixel)
    with open(args.output_csv,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({"units": "mm" if args.mm_per_pixel else "pixels",**summarize_track(rows)},indent=2))

if __name__=="__main__":main()
