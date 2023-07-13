import os, csv


fileext='.ngraph' #.ifcc.json
targetD = "D:\\iCallasm"
graphdir = "D:\\iCallasm\\original_ds"
logfile = "dsmapping.csv"
i = 0
with open(os.path.join(targetD, logfile), 'w') as f:
    for files in os.listdir(graphdir):
        filepath = os.path.join(graphdir, files)
        if os.path.isfile(filepath) and filepath.endswith(fileext):
            if os.path.exists(filepath[: -len(fileext)] + ".funcaddr2"):
                print("Renaming", i, filepath)
                f.write(str(i)+','+filepath+'\n')
                os.rename(filepath, os.path.join(targetD, str(i)+fileext))
                os.rename(filepath[: -len(fileext)] + ".funcaddr2", os.path.join(targetD, str(i)+".funcaddr2"))
                i += 1

'''
with open(os.path.join(directory, logfile), 'w') as f:
    for accounts in os.listdir(directory):
        account = os.path.join(directory, accounts)
        if os.path.isdir(account):
            for projects in os.listdir(account):
                project = os.path.join(account, projects)
                if os.path.isdir(project):
                    for binaries in os.listdir(project):
                        graphfile = os.path.join(project, binaries)
                        if os.path.isfile(graphfile) and graphfile.endswith('.bin'):
                            f.write(str(i)+','+graphfile+'\n')
                            os.rename(graphfile, os.path.join(targetD, str(i)+'.bin'))
                            i += 1
'''