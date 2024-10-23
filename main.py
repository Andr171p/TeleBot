from train import (
    train_model,
    predict_model,
    metrics,
    plots
)


def main() -> None:
    model = train_model()
    classifier = model.model()
    y_predict = predict_model(model=model)
    metrics(y_predict=y_predict)
    plots(model=classifier)


if __name__ == '__main__':
    main()
