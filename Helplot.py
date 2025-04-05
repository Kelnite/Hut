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
    metrics = self.hist.history[self.metrics]
    plt.plot(self.loop, metrics, label='Train Accuracy')
    self.labeler('Epochs', 'Accuracy', 'Model Result', True)
    plt.xticks(self.loop);

  @property
  def Falplot(self):
    """
    Model Error Plot
    """
    loss = self.hist.history['loss']
    plt.plot(self.loop, loss, label='Train Loss')
    self.labeler('Epochs', 'Loss', 'Model Error', True)
    plt.xticks(self.loop);