from .encoders import Encoder
from .decoders import Decoder
from .transformer import Transformer
from models.build import BuildModel

def build_base():
    en = Encoder(3,0)
    de = Decoder(10201,54,3,1)
    return Transformer(2,en,de)

BuildModel.add(0,build_base)
