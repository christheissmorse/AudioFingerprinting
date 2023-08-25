import numpy as np
import hashlib
import math
import cmath
import matplotlib.pyplot as plt
import wave
import warnings
from scipy.signal import find_peaks
import pickle
import os
import sys
import onnxruntime
import random
from sklearn.neighbors import NearestNeighbors
from PIL import Image
warnings.filterwarnings("ignore")
from IPython.display import Audio
from scipy.ndimage.filters import maximum_filter


# x = signal, w = window, h = hop length
def STFT(x, w, h):
  max_hops = (len(x) - len(w)) // h
  X = np.zeros(((len(w) // 2) + 1, max_hops), dtype=np.complex_)

  for n in range(0, max_hops):
    # clip of signal x; start at nth hop to the end of the window
    x_clip = x[n*h : n*h + len(w)]
      
    # plug output into n'th column of 2D output matrix
    col = np.fft.rfft(x_clip * w)
    X[:, n] = col
  return X


def generateSpectrogram(filepath, op = False):
  raw_signal = wave.open(filepath)
  nframes = raw_signal.getnframes()
  signal = np.frombuffer(raw_signal.readframes(nframes), dtype = np.int16)
  signal = signal.astype(float)
  framerate = raw_signal.getframerate()

  X = STFT(signal, np.hanning(756), 378)
  Y = np.abs(np.log(X)) ** 2

  #Y = np.flipud(Y)

  # fig, ax = plt.subplots()
  # ax.imshow(Y)
  # ax.set_xlabel('Time')
  # ax.set_ylabel('Frequency')
  # plt.gca().invert_yaxis()
  # plt.show()
  # plt.clf()

  return Y


def generateAllSpectrograms():
  for file in os.listdir("./audio_files/"):
      print(file)
      if '.wav' in file:
          Y = generateSpectrogram("./audio_files/" + file)
          # plt.xlabel("m")
          # plt.ylabel("k")
          # print(Y.shape)
          # plt.imshow(Y)
          im = Image.fromarray(Y).convert('L')
          im.save("./images/" + file[:-8] + ".png")



def extractKeypoints(data, bin_size, amp_thresh):
  detected_peaks = maximum_filter(data, size=bin_size)
  detected_peaks = np.where(detected_peaks == data, detected_peaks, 0)
  detected_peaks = np.where(detected_peaks < amp_thresh, 0, detected_peaks)

  # imm = Image.fromarray(detected_peaks).convert('L')
  # imm.show()

  coords = np.argwhere(detected_peaks!=0)
  f = coords[:,0]
  t = coords[:,1]

  # fig, ax = plt.subplots()
  # ax.imshow(data)
  # ax.scatter(t, f, color='orange', s = 6)
  # ax.set_xlabel('Time')
  # ax.set_ylabel('Frequency')
  # ax.set_title("bin_size: " + str(bin_size) + "; amp_thresh: " + str(amp_thresh))
  # plt.gca().invert_yaxis()
  # plt.show()
  # plt.clf()

  return coords


def extractFingerprint(data, coords, max_fingerprint_size):
  #fig, ax = plt.subplots()
  hashes = {}
  neighbors = None
  distances = None
  indicies = None

  try:
    neighbors = NearestNeighbors(n_neighbors = 10).fit(coords)
    distances, indicies = neighbors.kneighbors(coords)
  except:
    neighbors = NearestNeighbors(n_neighbors = 2).fit(coords)
    distances, indicies = neighbors.kneighbors(coords)
  
  for idxs in indicies:
    base_f = coords[idxs[0]][0]
    base_t = coords[idxs[0]][1]
    count = 0

    ids = []
    # returns base node and neighbors, so iterating over all the neighboring nodes here
    for i in range(1, len(idxs)):
      neighbor_f = coords[idxs[i]][0]
      neighbor_t = coords[idxs[i]][1]

      # compute time offset between base node and neighbors; keep only if in positive direction
      d_t = (neighbor_t - base_t)
      if d_t > 0:
        neighbor_id = "f" + str(neighbor_f) + "_" + "dt" + str(d_t)
        ids.append(neighbor_id)
        #plt.plot([base_t, neighbor_t], [base_f, neighbor_f], 'o', c = 'yellow', mfc = 'orange', mec = 'orange', linestyle="--", markersize = 3)
        count += 1
      if count == max_fingerprint_size:
        ids.sort()
        s1 = "f" + str(base_f) + ";"
        s2 = ";".join(ids)
        hashes[s1 + s2] = base_t
        break

  # ax.imshow(data)
  # ax.set_xlabel('Time')
  # ax.set_ylabel('Frequency')
  # ax.set_title("local_fingerprint_size: " + str(max_fingerprint_size + 1))
  # plt.gca().invert_yaxis()
  # plt.show()
  return hashes



def initDatabase(bin_size, amp_thresh, max_fingerprint_size):
  q = 0
  s = str(bin_size) + '-' + str(amp_thresh) + '-' + str(max_fingerprint_size) + '.pickle'
  with open(s, 'wb') as f:
    for file in os.listdir("./images/"):
      #print(q)
      if '.DS_Store' in file:
        continue
      im = Image.open("./images/" + file)
      data = np.asarray(im)

      coords = extractKeypoints(data, bin_size, amp_thresh)
      fp = extractFingerprint(data, coords, max_fingerprint_size)
      # ax.imshow(data)
      # ax.scatter(t, f, color='black', s = 6)
      # ax.set_xlabel('Time')
      # ax.set_ylabel('Frequency')
      # ax.set_title("Keypoint Detection")

      # plt.gca().invert_yaxis()
      # plt.show()
      # plt.clf()
      pickle.dump((file, fp), f)
      q += 1
      


def loadDatabase(path):
  with open(path, 'rb') as f:
    #To load from pickle file
    data = {}
    try:
        while True:
            d = pickle.load(f)
            data[d[0]] = d[1]

    except EOFError:
        pass

    return data



def test(database, bin_size, amp_thresh, max_fingerprint_size):
  accs = {}
  for noise_set in [0, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
    print("")
    print("----")
    print(noise_set)
    num_correct = 0
    total = 0
    for file in os.listdir("./testsets/" + str(noise_set)):
      if '.png' in file:

        # open noisy test spectrogram and get fingerprint
        im = Image.open("./testsets/" + str(noise_set) + '/' + file)
        data = np.asarray(im)
        kp = extractKeypoints(data, bin_size, amp_thresh)
        sample_fp = extractFingerprint(data, kp, max_fingerprint_size)
        # check against all fingerprints in database
        dd = {}
        for name in database:
          jj = 0
          dd[name] = {}
          d = {}
          fps = database[name]
          for s_fp in sample_fp:
            if s_fp in fps:
              jj += 1
              diff = fps[s_fp] - sample_fp[s_fp]
              if diff < 0:
                continue
              else:
                if diff in d:
                  d[diff] += 1
                else:
                  d[diff] = 1
          
          # calculate number of true local fingerprint matches
          score = 0
          if d != {}:
            score = max(d.values())
          dd[name] = score

        # predict based on totals
        prediction = max(dd, key=dd.get)
        if prediction == file:
          #print("CORRECT!")
          num_correct += 1
        # else:
        #   print("INCORRECT...")
        total += 1

    # report total accuracy
    total_accuracy = num_correct / total
    accs[str(noise_set)] = total_accuracy
    print(total_accuracy)
  return accs



def generateTestSet():
  np.random.seed(0)
  count = 0
  for file in os.listdir("./audio_files/"):
    if count == 4000:
      break
    if '.wav' in file:
      print(file)
      raw_signal = wave.open("./audio_files/" + file)
      nframes = raw_signal.getnframes()
      framerate = raw_signal.getframerate()
      signal = np.frombuffer(raw_signal.readframes(nframes), dtype = np.int16)
      signal = signal.astype(float)

      sdevs = [0, 0.01, 0.1, 1, 10, 100, 1000, 10000]
      for sdev in sdevs:
        # add noise to signal
        noise=np.random.normal(0, sdev, len(signal))
        noisy_signal = signal + noise

        # compute STFT
        X = STFT(noisy_signal, np.hanning(756), 378)
        Y = np.abs(np.log(X)) ** 2

        # get random 5 second clip from signal
        j = random.randint(0, Y.shape[1] - int((Y.shape[1]/6) - 1))
        Y = Y[:, j:j+int(Y.shape[1]/6)] # just 5 seconds

        #Y = np.flipud(Y)
        # fig, ax = plt.subplots()
        # ax.imshow(Y)
        # ax.set_xlabel('Time')
        # ax.set_ylabel('Frequency')
        # plt.gca().invert_yaxis()
        # plt.show()
        # plt.clf()

        im = Image.fromarray(Y).convert('L')
        im.save("./testsets/" + str(sdev) + '/' + file[:-8] + ".png")
      count += 1


def main():
  bin_size = int(sys.argv[1])
  amp_thresh = int(sys.argv[2])
  max_fingerprint_size = int(sys.argv[3])

  # convert audio files into spectrogram images
  generateAllSpectrograms()

  # extract fingerprints for database
  initDatabase(bin_size, amp_thresh, max_fingerprint_size)

  # load database
  db_hashes = loadDatabase(str(bin_size) + "-" + str(amp_thresh) + "-" + str(max_fingerprint_size) + ".pickle")
  
  # generate test set
  generateTestSet()

  # run tests
  accs = test(db_hashes, bin_size, amp_thresh, max_fingerprint_size)
  f = open(str(bin_size) + "-" + str(amp_thresh) + "-" + str(max_fingerprint_size) + ".txt", 'w')
  f.write(str(accs))

if __name__ == '__main__':
   main()