import os
import csv
import argparse

def main():
    parser = argparse.ArgumentParser(description="Split a restoration sample index by dataset name.")
    parser.add_argument("--sample-index", required=True, help="Path to completion-pair sample_index.csv")
    parser.add_argument("--output-dir", default=None, help="Directory for split CSV outputs")
    args = parser.parse_args()

    main_csv = args.sample_index
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(main_csv))
    os.makedirs(output_dir, exist_ok=True)
    bb_csv = os.path.join(output_dir, 'sample_index_bb.csv')
    fb_csv = os.path.join(output_dir, 'sample_index_fb.csv')
    
    with open(main_csv, 'r') as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        
        with open(bb_csv, 'w') as f_bb, open(fb_csv, 'w') as f_fb:
            writer_bb = csv.DictWriter(f_bb, fieldnames=fieldnames)
            writer_fb = csv.DictWriter(f_fb, fieldnames=fieldnames)
            
            writer_bb.writeheader()
            writer_fb.writeheader()
            
            for row in reader:
                if 'BreakingBad' in row['output_path']:
                    writer_bb.writerow(row)
                elif 'Fantastic' in row['output_path']:
                    writer_fb.writerow(row)
                    
    print(f"Saved split manifests to {output_dir}")

if __name__ == '__main__':
    main()
