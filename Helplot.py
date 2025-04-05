import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

class Helplot:
  """
  TensorFlow Model Training History Plot
  
  >>> history = model.fit(train, ...)
  >>> plot_hist = Helplot(history, 10)
  >>> plot_hist.Relplot
  >>> plot_hist.Falplot
  """
  def __init__(self, hist, loop, metrics):
    self.hist = hist
    self.loop = loop
    self.metrics = hist.history[metrics]

  @property
  def Relplot(self):
    """
    Model Result
    """
    plt.plot(self.loop, self.metrics, label='Train Accuracy')
    plt.title("Model Result")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks(self.loop)
    plt.legend();

  @property
  def Falplot(self):
    """
    Model Error Plot
    """
    plt.plot(self.loop)
    plt.title("Model Error")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(self.loop)
    plt.legend();
