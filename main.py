import torch
import torch.nn as nn
import io

# テスト対象のPyTorchレイヤを辞書に登録
def get_test_layers():
    return {
        "nn.Conv1d": (nn.Conv1d(3, 16, kernel_size=3), torch.randn(1, 3, 32)),
        "nn.Conv2d": (nn.Conv2d(3, 16, kernel_size=3), torch.randn(1, 3, 32, 32)),
        "nn.Conv3d": (nn.Conv3d(3, 16, kernel_size=3), torch.randn(1, 3, 32, 32, 32)),
        "nn.ConvTranspose1d": (nn.ConvTranspose1d(3, 16, kernel_size=3), torch.randn(1, 3, 32)),
        "nn.ConvTranspose2d": (nn.ConvTranspose2d(3, 16, kernel_size=3), torch.randn(1, 3, 32, 32)),
        "nn.ConvTranspose3d": (nn.ConvTranspose3d(3, 16, kernel_size=3), torch.randn(1, 3, 32, 32, 32)),
        "nn.LazyConv1d": (nn.LazyConv1d(16, kernel_size=3), torch.randn(1, 3, 32)),
        "nn.LazyConv2d": (nn.LazyConv2d(16, kernel_size=3), torch.randn(1, 3, 32, 32)),
        "nn.LazyConv3d": (nn.LazyConv3d(16, kernel_size=3), torch.randn(1, 3, 32, 32, 32)),
        "nn.LazyConvTranspose1d": (nn.LazyConvTranspose1d(16, kernel_size=3), torch.randn(1, 3, 32)),
        "nn.LazyConvTranspose2d": (nn.LazyConvTranspose2d(16, kernel_size=3), torch.randn(1, 3, 32, 32)),
        "nn.LazyConvTranspose3d": (nn.LazyConvTranspose3d(16, kernel_size=3), torch.randn(1, 3, 32, 32, 32)),
        "nn.Unfold": (nn.Unfold(kernel_size=3), torch.randn(1, 3, 32, 32)),
        "nn.Fold": (nn.Fold(output_size=(32, 32), kernel_size=3), torch.randn(1, 16, 32 * 32)),
        "nn.MaxPool1d": (nn.MaxPool1d(kernel_size=3), torch.randn(1, 3, 32)),
        "nn.MaxPool2d": (nn.MaxPool2d(kernel_size=3), torch.randn(1, 3, 32, 32)),
        "nn.MaxPool3d": (nn.MaxPool3d(kernel_size=3), torch.randn(1, 3, 32, 32, 32)),
        "nn.MaxUnpool1d": (nn.MaxUnpool1d(kernel_size=3), (torch.randn(1, 3, 32), torch.randint(0, 3, (1, 3, 32)))),
        "nn.MaxUnpool2d": (nn.MaxUnpool2d(kernel_size=3), (torch.randn(1, 3, 32, 32), torch.randint(0, 3, (1, 3, 32, 32)))),
        "nn.MaxUnpool3d": (nn.MaxUnpool3d(kernel_size=3), (torch.randn(1, 3, 32, 32, 32), torch.randint(0, 3, (1, 3, 32, 32, 32)))),
        "nn.AvgPool1d": (nn.AvgPool1d(kernel_size=3), torch.randn(1, 3, 32)),
        "nn.AvgPool2d": (nn.AvgPool2d(kernel_size=3), torch.randn(1, 3, 32, 32)),
        "nn.AvgPool3d": (nn.AvgPool3d(kernel_size=3), torch.randn(1, 3, 32, 32, 32)),
        "nn.FractionalMaxPool2d": (nn.FractionalMaxPool2d(kernel_size=3, output_size=(16, 16)), torch.randn(1, 3, 32, 32)),
        "nn.FractionalMaxPool3d": (nn.FractionalMaxPool3d(kernel_size=3, output_size=(16, 16, 16)), torch.randn(1, 3, 32, 32, 32)),
        "nn.LPPool1d": (nn.LPPool1d(2, kernel_size=3), torch.randn(1, 3, 32)),
        "nn.LPPool2d": (nn.LPPool2d(2, kernel_size=3), torch.randn(1, 3, 32, 32)),
        "nn.LPPool3d": (nn.LPPool3d(2, kernel_size=3), torch.randn(1, 3, 32, 32, 32)),
        "nn.AdaptiveMaxPool1d": (nn.AdaptiveMaxPool1d(output_size=16), torch.randn(1, 3, 32)),
        "nn.AdaptiveMaxPool2d": (nn.AdaptiveMaxPool2d(output_size=(16, 16)), torch.randn(1, 3, 32, 32)),
        "nn.AdaptiveMaxPool3d": (nn.AdaptiveMaxPool3d(output_size=(16, 16, 16)), torch.randn(1, 3, 32, 32, 32)),
        "nn.AdaptiveAvgPool1d": (nn.AdaptiveAvgPool1d(output_size=16), torch.randn(1, 3, 32)),
        "nn.AdaptiveAvgPool2d": (nn.AdaptiveAvgPool2d(output_size=(16, 16)), torch.randn(1, 3, 32, 32)),
        "nn.AdaptiveAvgPool3d": (nn.AdaptiveAvgPool3d(output_size=(16, 16, 16)), torch.randn(1, 3, 32, 32, 32)),
        "nn.ReflectionPad1d": (nn.ReflectionPad1d(1), torch.randn(1, 3, 32)),
        "nn.ReflectionPad2d": (nn.ReflectionPad2d(1), torch.randn(1, 3, 32, 32)),
        "nn.ReflectionPad3d": (nn.ReflectionPad3d(1), torch.randn(1, 3, 32, 32, 32)),
        "nn.ReplicationPad1d": (nn.ReplicationPad1d(1), torch.randn(1, 3, 32)),
        "nn.ReplicationPad2d": (nn.ReplicationPad2d(1), torch.randn(1, 3, 32, 32)),
        "nn.ReplicationPad3d": (nn.ReplicationPad3d(1), torch.randn(1, 3, 32, 32, 32)),
        "nn.ZeroPad1d": (nn.ConstantPad1d(1, 0), torch.randn(1, 3, 32)),
        "nn.ZeroPad2d": (nn.ConstantPad2d(1, 0), torch.randn(1, 3, 32, 32)),
        "nn.ZeroPad3d": (nn.ConstantPad3d(1, 0), torch.randn(1, 3, 32, 32, 32)),
        "nn.ConstantPad1d": (nn.ConstantPad1d(padding=2, value=0), torch.randn(1, 3, 32)),
        "nn.ConstantPad2d": (nn.ConstantPad2d(padding=2, value=0), torch.randn(1, 3, 32, 32)),
        "nn.ConstantPad3d": (nn.ConstantPad3d(padding=2, value=0), torch.randn(1, 3, 32, 32, 32)),
        "nn.CircularPad1d": (nn.ConstantPad1d(padding=2, value=0), torch.randn(1, 3, 32)),  # CircularPadは未サポートのため代用
        "nn.CircularPad2d": (nn.ConstantPad2d(padding=2, value=0), torch.randn(1, 3, 32, 32)),  # 同上
        "nn.CircularPad3d": (nn.ConstantPad3d(padding=2, value=0), torch.randn(1, 3, 32, 32, 32)),  # 同上
        "nn.ELU": (nn.ELU(), torch.randn(1, 3, 32, 32)),
        "nn.Hardshrink": (nn.Hardshrink(), torch.randn(1, 3, 32, 32)),
        "nn.Hardsigmoid": (nn.Hardsigmoid(), torch.randn(1, 3, 32, 32)),
        "nn.Hardtanh": (nn.Hardtanh(), torch.randn(1, 3, 32, 32)),
        "nn.Hardswish": (nn.Hardswish(), torch.randn(1, 3, 32, 32)),
        "nn.LeakyReLU": (nn.LeakyReLU(), torch.randn(1, 3, 32, 32)),
        "nn.LogSigmoid": (nn.LogSigmoid(), torch.randn(1, 3, 32, 32)),
        "nn.MultiheadAttention": (nn.MultiheadAttention(embed_dim=16, num_heads=2), (torch.randn(5, 1, 16), torch.randn(5, 1, 16), torch.randn(5, 1, 16))),
        "nn.PReLU": (nn.PReLU(), torch.randn(1, 3, 32, 32)),
        "nn.ReLU": (nn.ReLU(), torch.randn(1, 3, 32, 32)),
        "nn.ReLU6": (nn.ReLU6(), torch.randn(1, 3, 32, 32)),
        "nn.RReLU": (nn.RReLU(), torch.randn(1, 3, 32, 32)),
        "nn.SELU": (nn.SELU(), torch.randn(1, 3, 32, 32)),
        "nn.CELU": (nn.CELU(), torch.randn(1, 3, 32, 32)),
        "nn.GELU": (nn.GELU(), torch.randn(1, 3, 32, 32)),
        "nn.Sigmoid": (nn.Sigmoid(), torch.randn(1, 3, 32, 32)),
        "nn.SiLU": (nn.SiLU(), torch.randn(1, 3, 32, 32)),
        "nn.Mish": (nn.Mish(), torch.randn(1, 3, 32, 32)),
        "nn.Softplus": (nn.Softplus(), torch.randn(1, 3, 32, 32)),
        "nn.Softshrink": (nn.Softshrink(), torch.randn(1, 3, 32, 32)),
        "nn.Softsign": (nn.Softsign(), torch.randn(1, 3, 32, 32)),
        "nn.Tanh": (nn.Tanh(), torch.randn(1, 3, 32, 32)),
        "nn.Tanhshrink": (nn.Tanhshrink(), torch.randn(1, 3, 32, 32)),
        "nn.Threshold": (nn.Threshold(0.5, 0), torch.randn(1, 3, 32, 32)),
        "nn.GLU": (nn.GLU(), torch.randn(1, 3, 32, 32)),
        "nn.Softmin": (nn.Softmin(dim=1), torch.randn(1, 3, 32, 32)),
        "nn.Softmax": (nn.Softmax(dim=1), torch.randn(1, 3, 32, 32)),
        "nn.Softmax2d": (nn.Softmax2d(), torch.randn(1, 3, 32, 32)),
        "nn.LogSoftmax": (nn.LogSoftmax(dim=1), torch.randn(1, 3, 32, 32)),
        # "nn.AdaptiveLogSoftmaxWithLoss": (nn.AdaptiveLogSoftmaxWithLoss(10, 5, cutoffs=[5]), (torch.randn(1, 10), torch.randint(0, 5, (1,)))),
        "nn.BatchNorm1d": (nn.BatchNorm1d(3), torch.randn(1, 3, 32)),
        "nn.BatchNorm2d": (nn.BatchNorm2d(3), torch.randn(1, 3, 32, 32)),
        "nn.BatchNorm3d": (nn.BatchNorm3d(3), torch.randn(1, 3, 32, 32, 32)),
        "nn.LazyBatchNorm1d": (nn.LazyBatchNorm1d(), torch.randn(1, 3, 32)),
        "nn.LazyBatchNorm2d": (nn.LazyBatchNorm2d(), torch.randn(1, 3, 32, 32)),
        "nn.LazyBatchNorm3d": (nn.LazyBatchNorm3d(), torch.randn(1, 3, 32, 32, 32)),
        "nn.GroupNorm": (nn.GroupNorm(1, 3), torch.randn(1, 3, 32, 32)),
        "nn.SyncBatchNorm": (nn.SyncBatchNorm(3), torch.randn(1, 3, 32, 32)),
        "nn.InstanceNorm1d": (nn.InstanceNorm1d(3), torch.randn(1, 3, 32)),
        "nn.InstanceNorm2d": (nn.InstanceNorm2d(3), torch.randn(1, 3, 32, 32)),
        "nn.InstanceNorm3d": (nn.InstanceNorm3d(3), torch.randn(1, 3, 32, 32, 32)),
        "nn.LazyInstanceNorm1d": (nn.LazyInstanceNorm1d(), torch.randn(1, 3, 32)),
        "nn.LazyInstanceNorm2d": (nn.LazyInstanceNorm2d(), torch.randn(1, 3, 32, 32)),
        "nn.LazyInstanceNorm3d": (nn.LazyInstanceNorm3d(), torch.randn(1, 3, 32, 32, 32)),
        "nn.LayerNorm": (nn.LayerNorm([3, 32, 32]), torch.randn(1, 3, 32, 32)),
        "nn.LocalResponseNorm": (nn.LocalResponseNorm(size=5), torch.randn(1, 3, 32, 32)),
        "nn.RMSNorm": (nn.LayerNorm([3, 32, 32]), torch.randn(1, 3, 32, 32)),  # RMSNormはカスタム実装の必要あり
        "nn.RNN": (nn.RNN(input_size=10, hidden_size=20, num_layers=2), torch.randn(5, 3, 10)),
        "nn.LSTM": (nn.LSTM(input_size=10, hidden_size=20, num_layers=2), torch.randn(5, 3, 10)),
        "nn.GRU": (nn.GRU(input_size=10, hidden_size=20, num_layers=2), torch.randn(5, 3, 10)),
        "nn.RNNCell": (nn.RNNCell(input_size=10, hidden_size=20), torch.randn(3, 10)),
        "nn.LSTMCell": (nn.LSTMCell(input_size=10, hidden_size=20), torch.randn(3, 10)),
        "nn.GRUCell": (nn.GRUCell(input_size=10, hidden_size=20), torch.randn(3, 10)),
        "nn.Transformer": (nn.Transformer(d_model=16, nhead=2), (torch.randn(10, 32, 16), torch.randn(10, 32, 16))),
        "nn.TransformerEncoder": (nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=16, nhead=2), num_layers=2), torch.randn(10, 32, 16)),
        "nn.TransformerDecoder": (nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=16, nhead=2), num_layers=2), (torch.randn(10, 32, 16), torch.randn(10, 32, 16))),
        "nn.TransformerEncoderLayer": (nn.TransformerEncoderLayer(d_model=16, nhead=2), torch.randn(10, 32, 16)),
        "nn.TransformerDecoderLayer": (nn.TransformerDecoderLayer(d_model=16, nhead=2), (torch.randn(10, 32, 16), torch.randn(10, 32, 16))),
        "nn.Identity": (nn.Identity(), torch.randn(1, 3, 32, 32)),
        "nn.Linear": (nn.Linear(10, 5), torch.randn(1, 10)),
        "nn.Bilinear": (nn.Bilinear(10, 10, 5), (torch.randn(1, 10), torch.randn(1, 10))),
        "nn.LazyLinear": (nn.LazyLinear(5), torch.randn(1, 10)),
        "nn.Dropout": (nn.Dropout(), torch.randn(1, 3, 32, 32)),
        "nn.Dropout1d": (nn.Dropout1d(), torch.randn(1, 3, 32)),
        "nn.Dropout2d": (nn.Dropout2d(), torch.randn(1, 3, 32, 32)),
        "nn.Dropout3d": (nn.Dropout3d(), torch.randn(1, 3, 32, 32, 32)),
        "nn.AlphaDropout": (nn.AlphaDropout(), torch.randn(1, 3, 32, 32)),
        "nn.FeatureAlphaDropout": (nn.FeatureAlphaDropout(), torch.randn(1, 3, 32, 32)),
        "nn.Embedding": (nn.Embedding(10, 3), torch.randint(0, 10, (1, 5))),
        "nn.EmbeddingBag": (nn.EmbeddingBag(10, 3), torch.randint(0, 10, (1, 5))),
        "nn.PixelShuffle": (nn.PixelShuffle(2), torch.randn(1, 12, 16, 16)),
        "nn.PixelUnshuffle": (nn.PixelUnshuffle(2), torch.randn(1, 3, 32, 32)),
        "nn.Upsample": (nn.Upsample(scale_factor=2, mode='nearest'), torch.randn(1, 3, 16, 16)),
        "nn.UpsamplingNearest2d": (nn.Upsample(scale_factor=2, mode='nearest'), torch.randn(1, 3, 16, 16)),
        "nn.UpsamplingBilinear2d": (nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), torch.randn(1, 3, 16, 16)),
        # "nn.ChannelShuffle": ("未対応", None),  # PyTorchに標準では存在しないレイヤ
        "nn.DataParallel": (nn.DataParallel(nn.Linear(10, 5)), torch.randn(1, 10)),
        # "nn.parallel.DistributedDataParallel": (nn.parallel.DistributedDataParallel(nn.Linear(10, 5)), torch.randn(1, 10)),
        # 他のレイヤもこの形式で追加可能
    }

# ONNXエクスポートを試行する関数
def check_onnx_compatibility(layer_name, layer, dummy_input):
    model = nn.Sequential(layer)
    # f = io.BytesIO()  # メモリバッファを使用
    f = layer_name + ".onnx"
    try:
        torch.onnx.export(
            model,                       # モデル
            dummy_input,                 # ダミー入力
            f,                           # 出力先
            opset_version=11,            # ONNX Opsetバージョン
            verbose=True                # 詳細ログを抑制
        )
        print(f"[OK] {layer_name} can be converted to ONNX")
    except Exception as e:
        print(f"[NG] {layer_name} can't be converted to ONNX: {e}")

# 各レイヤをチェック
test_layers = get_test_layers()
for layer_name, (layer, dummy_input) in test_layers.items():
    check_onnx_compatibility(layer_name, layer, dummy_input)

