from getpwnam import pwd
from getgrnam import grp
import os

uid = getpwnam("YOUR_USERNAME")[2]
gid = grp.getgrnam("YOUR_GROUPNAME")[2]
os.chown("myPath/xFiles.bin.addr_patched", uid, gid)