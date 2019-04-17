#!/bin/bash

# get the filename without the last extention (will give back FILE.tar if original was FILE.tar.gz
echo "${FILE%.*}"

# too many spaces in downloaded data filenames to do this automatically
# skipping the cell cycle files
# and the storm overlap file


[arrajpur@sh-ln03 login ~/Analysis_Project]$ clean_data.py downloaded_data/HCT116_chr21-28-30Mb_6h\ auxin.csv clean_data/HCT116_chr21-28-30Mb_6h_auxin.tsv
[arrajpur@sh-ln03 login ~/Analysis_Project]$ clean_data.py downloaded_data/HCT116_chr21-28-30Mb_untreated.csv clean_data/HCT116_chr21-28-30Mb_untreated.tsv
[arrajpur@sh-ln03 login ~/Analysis_Project]$ clean_data.py downloaded_data/HCT116_chr21-34-37Mb_6h\ auxin.csv clean_data/HCT116_chr21-34-37Mb_6h_auxin.tsv
[arrajpur@sh-ln03 login ~/Analysis_Project]$ clean_data.py downloaded_data/HCT116_chr21-34-37Mb_untreated.csv clean_data/HCT116_chr21-34-37Mb_untreated.tsv
[arrajpur@sh-ln03 login ~/Analysis_Project]$ clean_data.py downloaded_data/IMR90_chr21-18-20Mb.csv clean_data/IMR90_chr21-18-20Mb.tsv
[arrajpur@sh-ln03 login ~/Analysis_Project]$ clean_data.py downloaded_data/IMR90_chr21-28-30Mb.csv clean_data/IMR90_chr21-28-30Mb.tsv
[arrajpur@sh-ln03 login ~/Analysis_Project]$ clean_data.py downloaded_data/K562_chr21-28-30Mb.csv clean_data/K562_chr21-28-30Mb.tsv

[arrajpur@sh-ln03 login ~/Analysis_Project/clean_data]$ ls
A549_chr21-28-30Mb.tsv              HCT116_chr21-34-37Mb_untreated.tsv
HCT116_chr21-28-30Mb_6h_auxin.tsv   IMR90_chr21-18-20Mb.tsv
HCT116_chr21-28-30Mb_untreated.tsv  IMR90_chr21-28-30Mb.tsv
HCT116_chr21-34-37Mb_6h_auxin.tsv   K562_chr21-28-30Mb.tsv

