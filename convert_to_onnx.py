import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import tensorflow as tf
import tf2onnx

BASE  = Path(__file__).parent
ruta  = BASE / 'models' / 'chiari_DenseNet121_kfold.keras'
salida = BASE / 'models' / 'chiari_DenseNet121_kfold.onnx'

print(f'Cargando modelo desde: {ruta}')
modelo = tf.keras.models.load_model(str(ruta))

print('Convirtiendo a ONNX...')
input_sig = [tf.TensorSpec([1, 224, 224, 3], tf.float32, name='input')]
onnx_model, _ = tf2onnx.convert.from_keras(modelo, input_signature=input_sig, opset=13)

with open(salida, 'wb') as f:
    f.write(onnx_model.SerializeToString())

size_mb = salida.stat().st_size / 1e6
print(f'Guardado: {salida}')
print(f'Tamano  : {size_mb:.1f} MB')
