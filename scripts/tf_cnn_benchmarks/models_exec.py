import subprocess

#   subprocess.call(["ls", "-l"])
models = ['inception3', 'vgg16', 'alexnet']
gpus = [1, 2]
batch_size = [1,8,16,32,64,128, 256, 512, 640, 712, 824]

for model in models:
    for gpu in gpus:
        for bs in batch_size:
            GPU =  "--num_gpus={}".format(gpu)
            BS = "--batch_size={}".format(bs)
            MODEL = "--model={}".format(model)
            file = "./models_output/{}/{}_{}gpu_{}bs.txt".format(model, model, gpu, bs)
            f = open(file,'wb')
            subprocess.call(["python", "tf_cnn_benchmarks.py", GPU, BS, "--forward_only=true", MODEL, "--variable_update=parameter_server"],stdout=f)

    print("COMPLETED {}".format(model))

print("-----------COMPLETED----------------------")

