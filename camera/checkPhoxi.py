from harvesters.core import Harvester
from sys import platform
import os
import copy
def checkPhoxi():
    if platform == "win32":
        cti_file_path_suffix = "/API/bin/photoneo.cti"
    else:
        cti_file_path_suffix = "/API/lib/photoneo.cti"
    cti_file_path = os.getenv('PHOXI_CONTROL_PATH') + cti_file_path_suffix
    with Harvester() as h:
        h.add_file(cti_file_path, True, True)
        h.update()
        device_list = list(h.device_info_list)
    return True if len(device_list) > 0 else False

