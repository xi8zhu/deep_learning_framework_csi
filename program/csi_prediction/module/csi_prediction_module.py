# maybe we can use env  ---finished



model_list = ['ConvLSTM']
def choose_module(cfg):
    """
    input: model name
    output: model
    """
    assert cfg.name in model_list, "Ineffective model"
    if cfg.name == 'ConvLSTM':
        from ConvLSTM import MyConvLSTM
        return MyConvLSTM()

    elif cfg.name == 'LSTM':
        import LSTM
    elif cfg.name == 'CNN':
        import CNN