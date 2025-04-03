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
    self.metrics = metrics

  @property
  def Relplot(self):
    """
    """
    if self.val_log in self.hist:
      plt.plot(self.loop, self.val_log)
    plt.title("Model Result")
    plt.legend();

  @property
  def Falplot(self):
    """
    Model Error Plot
    """
    plt.plot(self.loop)
    if self.val_loss in self.hist:
      plt.plot(self.loop, self.val_loss)
    plt.title("Model Error")
    plt.legend();
