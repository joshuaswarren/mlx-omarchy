import json, numpy as np, mlx.core as mx
mx.set_default_device(mx.gpu)
x=mx.split(mx.array([[1.,2.],[3.,4.]],dtype=mx.float32),2,axis=1)[0]
y=mx.log(x); mx.eval(y); actual=np.array(y); expected=np.log(np.array([[1.],[3.]],dtype=np.float64)).astype(np.float32)
print(json.dumps({"version":mx.__version__,"device":str(mx.default_device()),"actual":actual.tolist(),"expected":expected.tolist(),"actual_bits":actual.view(np.uint32).tolist(),"expected_bits":expected.view(np.uint32).tolist()}))
