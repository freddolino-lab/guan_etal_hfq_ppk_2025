import argparse
import gzip
from collections import defaultdict
import pysam


def count_initial_as(sequence):
    count = 0
    for base in sequence:
        if base == 'T':
            count += 1
        else:
            break
    return count

def parse_fastq(fastq_file):
    a_count_dict = {}
    with gzip.open(fastq_file, 'rt') as f:
        while True:
            header = f.readline().strip()
            if not header:
                break  # EOF
            seq = f.readline().strip()
            f.readline()  # Plus line
            qual = f.readline().strip()  # Quality line

            read_name = header.split()[0][1:]
            a_count_dict[read_name] = count_initial_as(seq)
    return a_count_dict

def main(fastq_file, bam_file, output_bed):
    # Parse the fastq file to get count of 'T' residues at beginning of each read
    a_count_dict = parse_fastq(fastq_file)
    
    # Open the bam file and read alignments
    bamfile = pysam.AlignmentFile(bam_file, "rb")
    
    # Prepare output in bed format
    with open(output_bed, 'w') as bedfile:
        for read in bamfile.fetch():
            read_name = read.query_name

            if not(read.is_read1):
                continue

            if read_name in a_count_dict:
                a_count = a_count_dict[read_name]
                chrom = read.reference_name
                start = read.reference_start
                end = read.reference_end
                strand = '+' if not read.is_reverse else '-'
                
                bedfile.write(f"{chrom}\t{start}\t{end}\t{read_name}\t{a_count}\t{strand}\n")
    
    bamfile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process FASTQ and BAM files to output a BED format file")
    parser.add_argument("fastq_file", help="Path to the gzipped FASTQ file")
    parser.add_argument("bam_file", help="Path to the BAM file")
    parser.add_argument("output_bed", help="Path to the output BED file")
    
    args = parser.parse_args()
    
    main(args.fastq_file, args.bam_file, args.output_bed)
