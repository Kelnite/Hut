import os, shutil, urrlib

def GetFile(url, filename, filepath, unpack=False, file_type="zip"):
  """
  Get Your File, Optionally Open !
  """
  retrieve = os.path.join(url, filename)
  urllib.request.urlretrieve(retrieve, filename)
  if unpack:
    shutil.unpack_archive(filename, filepath, file_type)
  return filename if not unpack else filepath