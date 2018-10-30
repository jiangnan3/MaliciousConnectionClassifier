import pandas

dos = ['back.', 'land.', 'neptune.', 'pod.', 'smurf.', 'teardrop.']
u2r = ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.']
r21 = ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'phf.', 'spy.', 'warezclient.', 'warezmaster.']
probe = ['ipsweep.', 'nmap.', 'portsweep.', 'satan.']
normal = ['normal.']

protocol_type = ['udp', 'icmp', 'tcp']
service = ['domain', 'netbios_ssn', 'urp_i', 'Z39_50', 'smtp', 'gopher', 'private', 'echo', 'printer', 'red_i',
           'eco_i', 'sunrpc', 'ftp_data', 'urh_i', 'pm_dump', 'pop_3', 'pop_2', 'systat', 'ftp', 'uucp', 'whois',
           'netbios_dgm', 'efs', 'remote_job', 'sql_net', 'daytime', 'ntp_u', 'finger', 'ldap', 'netbios_ns',
           'kshell', 'iso_tsap', 'ecr_i', 'nntp', 'shell', 'domain_u', 'uucp_path', 'courier', 'exec', 'tim_i',
           'netstat', 'telnet', 'rje', 'hostnames', 'link', 'auth', 'http_443', 'csnet_ns', 'X11', 'IRC', 'tftp_u',
           'imap4', 'supdup', 'name', 'nnsp', 'mtp', 'http', 'bgp', 'ctf', 'klogin', 'vmnet', 'time', 'discard',
           'login', 'other', 'ssh']
flag = ['OTH', 'RSTR', 'S3', 'S2', 'S1', 'S0', 'RSTOS0', 'REJ', 'SH', 'RSTO', 'SF']

data = pandas.read_csv("kddcup.data", header=None)

for item in protocol_type:
    data[1][data[1].isin([item])] = protocol_type.index(item)
for item in service:
    data[2][data[2].isin([item])] = service.index(item)
for item in flag:
    data[3][data[3].isin([item])] = flag.index(item)


data[41][data[41].isin(dos)] = 0
data[41][data[41].isin(u2r)] = 1
data[41][data[41].isin(r21)] = 2
data[41][data[41].isin(probe)] = 3
data[41][data[41].isin(normal)] = 4

data.to_csv("datacopy", index=False, header=False)