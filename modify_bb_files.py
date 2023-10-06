file1=open("C:/Users/vatan/Downloads/group1_4_changed.txt", "w")
file2=open("C:/Users/vatan/Downloads/groundtruthgroup1_changed.txt", "w")

with open("C:/Users/vatan/Downloads/group1_4.txt") as f:
    for line in f:
        string=line.split(",")
        string[3],string[4]="84","84"
        new_line=",".join(string)
        file1.write(new_line)

with open("C:/Users/vatan/Downloads/groundtruthgroup1.txt") as f:
    for line in f:
        string=line.split(",")
        string[2],string[3]="84","84"
        new_line=",".join(string)
        file2.write(new_line+"\n")
