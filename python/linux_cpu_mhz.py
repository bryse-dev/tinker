import re
import pprint
import time

def get_mhz():
    cpu_mhz_dict = {}
    p = re.compile('[0-9\.]+')
    with open('/proc/cpuinfo', 'r') as fh:
        processor = None
        mhz = None
        for line in fh.readlines():
          if "processor" in line:
            processor = p.findall(line)[0]
          if "cpu MHz" in line:
            mhz = p.findall(line)[0]
            cpu_mhz_dict[processor] = mhz
    return cpu_mhz_dict

while True:
    pprint.pprint(get_mhz())
    time.sleep(1)
