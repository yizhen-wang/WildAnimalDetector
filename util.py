def writeFile(f, data):
	f = open(f, 'a')
	f.write(str(data) + '\n')
	f.close()

def load_file(f):
	info_list = []
	file = open(f)
	while True:
		line = file.readline().strip()
		if not line or line == "":
			break
		else:
			info_list.append(line)
	return info_list

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_parameter_number(net):
	total_num = sum(p.numel() for p in net.parameters())
	trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
	return {'Total': total_num, 'Trainable': trainable_num}

def epoch_time(start_time, end_time):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time / 60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins, elapsed_secs
