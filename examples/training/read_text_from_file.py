data_fname = 'real_data.txt'
data_file = open(data_fname,"r") 
lines = data_file.readlines() 
data_file.close() 
print(lines)