import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

class Helplot:
  """
  TensorFlow Model Training History Plot
  
  >>> history = model.fit(train, ...)
  >>> plot_hist = Helplot(history, 'accuracy')
  >>> plot_hist.Relplot
  >>> plot_hist.Falplot
  """
  def __init__(self, hist, metrics):
    self.hist = hist.history
    self.loop = [*range(1, len(self.hist['loss']) + 1)]
    self.metrics = metrics

  def labeler(self, xlabel, ylabel, title, legend=False):
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    if legend:
      plt.legend()
    plt.title(title)

  @property
  def Relplot(self):
    """
    Model Result
    """
    result = self.hist[self.metrics]
    plt.plot(self.loop, result, label='Train Accuracy')
    if 'val_' + self.metrics in self.hist.keys():
      val_result = self.hist['val_' + self.metrics]
      plt.plot(self.loop, val_result, label='Validation Accuracy')
    self.labeler('Epochs', 'Accuracy', 'Model Result', True)
    plt.xticks(self.loop);

  @property
  def Falplot(self):
    """
    Model Error Plot
    """
    loss = self.hist['loss']
    plt.plot(self.loop, loss, label='Train Loss')
    if 'val_loss' in self.hist.keys():
      val_loss = self.hist['val_loss']
      plt.plot(self.loop, val_loss, label='Validation Loss')
    self.labeler('Epochs', 'Loss', 'Model Error', True)
    plt.xticks(self.loop);
