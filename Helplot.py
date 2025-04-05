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
    self.loss = hist.history['loss']

  def labeler(self, xlabel, ylabel, title, legend=False):
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    if legend:
      plt.legend()

  @property
  def Relplot(self):
    """
    Model Result
    """
    plt.plot(self.loop, self.metrics, label='Train Accuracy')
    self.labeler('Epochs', 'Accuracy', 'Model Result', True)
    plt.xticks(self.loop);

  @property
  def Falplot(self):
    """
    Model Error Plot
    """
    plt.plot(self.loop, self.loss, label='Train Loss')
    self.labeler('Epochs', 'Loss', 'Model Error', True)
    plt.xticks(self.loop);