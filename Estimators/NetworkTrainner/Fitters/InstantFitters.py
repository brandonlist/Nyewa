from Estimators.NetworkTrainner.Fitters.BaseFitters import InstantBaseFitter

class InstantFitter(InstantBaseFitter):
    def __init__(self,
                 model,
                 train_set,
                 valid_set,
                 test_set,
                 metric='acc'):
        super(InstantFitter, self).__init__(model=model,
                                            train_set=train_set,
                                            valid_set=valid_set,
                                            test_set=test_set,
                                            metric=metric)

    def fit_train_data_on_model(self):
        X,y = self.datasets['train'].X , self.datasets['train'].y
        self.model.fit(X,y)

    def test_data_score_on_model(self):
        X,y = self.datasets['test'].X, self.datasets['test'].y
        acc = self.model.score(X,y)
        return acc

    def run(self):
        self.fit_train_data_on_model()
        acc = self.test_data_score_on_model()
        if self.metric=='acc':
            return acc