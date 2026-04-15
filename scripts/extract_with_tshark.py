# Copyright © 2026 Sonomos, Inc.
# All rights reserved.
import argparse, csv, json, subprocess, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import FEATURE_NAMES, NUM_FEATURES, sni_ngram_hash, _HASH_DIMS

def find_tshark():
    for p in [r"C:\Program Files\Wireshark\tshark.exe", r"C:\Program Files (x86)\Wireshark\tshark.exe", "tshark"]:
        try:
            r = subprocess.run([p, "--version"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0: return p
        except: pass
    print("ERROR: tshark not found"); sys.exit(1)

def extract_flows(pcap, tshark):
    print(f"  Extracting flows with tshark...")
    cmd = [tshark, "-r", pcap, "-T", "fields", "-e", "tcp.stream", "-e", "frame.time_epoch", "-e", "frame.len", "-e", "tcp.srcport", "-e", "tcp.dstport", "-e", "ip.src", "-e", "ip.dst", "-e", "tls.handshake.extensions_server_name", "-E", "separator=|", "-E", "header=n", "-Y", "tcp"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  tshark error: {result.stderr[:200]}"); return []
    streams = defaultdict(lambda: {"packets":[],"timestamps":[],"src_port":None,"dst_port":None,"src_ip":None,"dst_ip":None,"sni":""})
    for line in result.stdout.strip().split("\n"):
        if not line.strip(): continue
        p = line.split("|")
        if len(p) < 7: continue
        sid = p[0].strip()
        if not sid: continue
        try: ts=float(p[1].strip()); pl=int(p[2].strip())
        except: continue
        s = streams[sid]
        s["packets"].append(pl); s["timestamps"].append(ts)
        if s["src_port"] is None: s["src_port"]=p[3].strip(); s["dst_port"]=p[4].strip(); s["src_ip"]=p[5].strip(); s["dst_ip"]=p[6].strip()
        sni = p[7].strip() if len(p)>7 else ""
        if sni and not s["sni"]: s["sni"]=sni
    print(f"  Found {len(streams)} TCP streams")
    flows = []
    for sid, s in streams.items():
        pkts=s["packets"]; times=s["timestamps"]
        if len(pkts)<3: continue
        up_s=[pkts[i] for i in range(0,len(pkts),2)]; dn_s=[pkts[i] for i in range(1,len(pkts),2)]
        iats=[times[i]-times[i-1] for i in range(1,len(times)) if 0<=times[i]-times[i-1]<30]
        dur=max(times[-1]-times[0],0.001) if len(times)>1 else 0.001
        flows.append({"packet_sizes":pkts,"upstream_sizes":up_s,"downstream_sizes":dn_s,"upstream_bytes":sum(up_s),"downstream_bytes":sum(dn_s),"iats":iats,"duration":dur,"n_upstream":len(up_s),"n_downstream":len(dn_s),"total_bytes":sum(pkts),"sni":s["sni"],"first_n":pkts[:8]})
    print(f"  Kept {len(flows)} flows")
    return flows

def flow_to_features(f):
    feat = np.zeros(NUM_FEATURES, dtype=np.float32)
    pkts=f["packet_sizes"]; iats=f["iats"]; dur=f["duration"]
    if pkts:
        a=np.array(pkts,dtype=np.float32); feat[0]=np.mean(a)/1500; feat[1]=np.std(a)/1500; feat[2]=np.min(a)/1500; feat[3]=np.max(a)/1500; feat[4]=np.percentile(a,25)/1500; feat[5]=np.percentile(a,50)/1500; feat[6]=np.percentile(a,75)/1500
    if iats:
        a=np.array(iats,dtype=np.float32); feat[7]=min(np.mean(a),10)/10; feat[8]=min(np.std(a),10)/10; feat[9]=min(np.min(a),10)/10; feat[10]=min(np.max(a),10)/10; feat[11]=min(np.median(a),10)/10
    feat[12]=np.log1p(min(dur,300))/np.log1p(300); feat[13]=np.log1p(f["n_upstream"])/np.log1p(10000); feat[14]=np.log1p(f["n_downstream"])/np.log1p(10000); feat[15]=np.log1p(f["total_bytes"]/dur)/np.log1p(1e9)
    up=f["upstream_sizes"]; dn=f["downstream_sizes"]
    if up: a=np.array(up,dtype=np.float32); feat[16]=np.mean(a)/1500; feat[17]=np.std(a)/1500; feat[18]=np.median(a)/1500
    if dn: a=np.array(dn,dtype=np.float32); feat[19]=np.mean(a)/1500; feat[20]=np.std(a)/1500; feat[21]=np.median(a)/1500
    td=f["upstream_bytes"]+f["downstream_bytes"]; feat[22]=f["upstream_bytes"]/td if td>0 else 0.5
    tp=f["n_upstream"]+f["n_downstream"]; feat[23]=f["n_upstream"]/tp if tp>0 else 0.5
    for i,s in enumerate(f["first_n"][:8]): feat[24+i]=s/1500
    feat[32]=1.0; feat[33]=0.5; feat[34]=0.4; feat[37]=1.0; feat[39]=1.0; feat[44]=1.0
    if f["sni"]: feat[50:50+_HASH_DIMS]=sni_ngram_hash(f["sni"],dims=_HASH_DIMS)
    return feat

def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--pcap", type=str); g.add_argument("--pcap-dir", type=str)
    parser.add_argument("--label", type=int, choices=[0,1], required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    tshark = find_tshark(); print(f"Using tshark: {tshark}")
    all_X, all_y = [], []
    pcaps = [Path(args.pcap)] if args.pcap else sorted(list(Path(args.pcap_dir).glob("*.pcap"))+list(Path(args.pcap_dir).glob("*.pcapng")))
    if not pcaps: print("No pcap files found"); sys.exit(1)
    for pf in pcaps:
        print(f"\nProcessing {pf.name} (label={args.label})...")
        flows = extract_flows(str(pf), tshark)
        if not flows: continue
        X = np.stack([flow_to_features(f) for f in flows]); y = np.full(len(X), args.label, dtype=np.float32)
        snis = set(f["sni"] for f in flows if f["sni"])
        if snis: print(f"  SNI domains: {', '.join(sorted(snis)[:10])}")
        all_X.append(X); all_y.append(y)
    if not all_X: print("No flows extracted"); sys.exit(1)
    X_full=np.vstack(all_X); y_full=np.concatenate(all_y)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output,"w",newline="") as f:
        w=csv.writer(f); w.writerow(FEATURE_NAMES+["label"])
        for i in range(len(X_full)): w.writerow([f"{v:.6f}" for v in X_full[i]]+[f"{y_full[i]:.0f}"])
    print(f"\nSaved to {args.output}"); print(f"  Total flows: {len(y_full)}, AI: {int(y_full.sum())}, Normal: {int(len(y_full)-y_full.sum())}")

if __name__ == "__main__": main()
