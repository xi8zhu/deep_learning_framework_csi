# maybe we can use env  ---finished



model_list = ['ConvLSTM', 'TransLSTM']
def choose_module(cfg):
    """
    input: model name
    output: model
    """
    model_name = cfg.model_name
    assert model_name in model_list, "Ineffective model"
    if model_name == 'ConvLSTM':
        from ConvLSTM import MyConvLSTM
        return MyConvLSTM()
    elif model_name == 'LSTM':
        from LSTM import LSTM
        return LSTM()
    elif model_name == 'CNN':
        from CNN import CNN
        return CNN()
    elif model_name == 'TransLSTM':
        from TransLSTM import TransLSTM
        return TransLSTM()