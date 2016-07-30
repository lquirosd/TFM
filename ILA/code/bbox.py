import scipy.ndimage as ndi

class BBox(object):
   def __init__(self, x1, y1, x2, y2):
      '''
      (x1, y1) is the upper left corner,
      (x2, y2) is the lower right corner,
      with (0, 0) being in the upper left corner.
      '''
      if x1 > x2: x1, x2 = x2, x1
      if y1 > y2: y1, y2 = y2, y1
      self.x1 = x1
      self.y1 = y1
      self.x2 = x2
      self.y2 = y2
      self.area = (x2 - x1) * (y2 - y1)

   def taxicab_diagonal(self):
      '''
      Return the taxicab distance from (x1,y1) to (x2,y2)
      '''
      return self.x2 - self.x1 + self.y2 - self.y1
   def overlaps(self, other):
      '''
      Return True iff self and other overlap.
      '''
      return not ((self.x1 > other.x2)
                  or (self.x2 < other.x1)
                  or (self.y1 > other.y2)
                  or (self.y2 < other.y1))
   def __eq__(self, other):
      return (self.x1 == other.x1
               and self.y1 == other.y1
               and self.x2 == other.x2
               and self.y2 == other.y2)

def find_paws(data, smooth_radius = 5, threshold = 0.0001):
   # http://stackoverflow.com/questions/4087919/how-can-i-improve-my-paw-detection
   """Detects and isolates contiguous regions in the input array"""
   # Blur the input data a bit so the paws have a continous footprint 
   data = ndi.uniform_filter(data, smooth_radius)
   # Threshold the blurred data (this needs to be a bit > 0 due to the blur)
   thresh = data < threshold
   # Fill any interior holes in the paws to get cleaner regions...
   filled = ndi.morphology.binary_fill_holes(thresh)
   # Label each contiguous paw
   coded_paws, num_paws = ndi.label(filled)
   # Isolate the extent of each paw
   # find_objects returns a list of 2-tuples: (slice(...), slice(...))
   # which represents a rectangular box around the object
   data_slices = ndi.find_objects(coded_paws)
   return data_slices

def slice_to_bbox(slices):
   for s in slices:
      dy, dx = s[:2]
      yield BBox(dx.start, dy.start, dx.stop+1, dy.stop+1)

