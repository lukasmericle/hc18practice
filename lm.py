def to_submission(chrom, title):
    rs = to_rectangles(chrom)
    n = rs.shape[1]
    f = open("submission_"+title+".txt")
    f.write(str(n)+'\n')
    for r in rs:
        r1 = r[0]; c1 = r[1]; r2 = r[2]; c2 = r[3]
        f.write(str(r[0]) + ' ' + str(r[1]) + ' ' + str(r2) + ' ' + str(c2) + '\n')
    f.close()