import os
import subprocess
def lecture_configs(file_path:str):
    f=open(file_path,'r')
    content=f.read()
    f.close()
    lines=content.split('\n')
    return lines

def lecture_ligne(line:str,d_out:dict,existing_modes:list)->dict:
    elements=line.split(";")
    for e in elements:
        if len(e)>1:
            name_param,val_param=e.split("=")
            name_param=name_param.replace(" ","")
            val_param=val_param.replace(" ","")
            if name_param not in d_out.keys() or (name_param=="mode" and val_param not in existing_modes):
                print(f"{e} non reconnu dans les fichier --> exit")
                exit()
            d_out[name_param]=val_param
    return d_out
        
def modif_param_simu(file_path:str,config_to_launch:dict):
    replace=""
    lines_to_mod=list(config_to_launch.keys())
    f=open(file_path,'r')
    lines=f.readlines()
    f.close()
    for line in lines:
        was_replaced=False
        for to_mod in lines_to_mod:
            if to_mod+"=" in line.strip().replace(" ",""):
                replace+=to_mod+"="+str(config_to_launch[to_mod])+"\n"
                was_replaced=True
        if len(line)>1 and not was_replaced:
            replace+=line
    f=open(file_path,'w')
    f.write(replace)
    f.close()

script_dir = os.path.dirname(os.path.abspath(__file__))
existing_modes=["singlet-triplet","singlet-singlet"]

lines=lecture_configs(f"{script_dir}\\configs_to_run.config")
L_configs=[]
dico={"code_to_launch":0,"TARGET_NU":0,"TARGET_NT":0,"delta_U_vals_full":0,"delta_t_vals_full":0,"psi0_label":0,"init_sig":0,"psi0":0}
for line in lines:
    L_configs.append(lecture_ligne(line,dico,existing_modes))
    if(L_configs[-1]["code_to_launch"]==0):
        print("missing file name")
        exit()
    modif_param_simu(script_dir+"\\param_simu.py",dico)
    subprocess.call(["python",script_dir+"\\"+dico["code_to_launch"].replace("\"","")])
