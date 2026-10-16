import sys,json
sys.path.insert(0,'src2d');import search_hysteresis15 as s
if __name__=='__main__':
 keys=['5a072d31d56b3b40ee8c','777f493b0800c82fcc45','6638d569b7d942f72270']
 s.batch([dict(json.loads((s.OUT/(key+'.json')).read_text())['config'],seconds=3000.,dt=.00025) for key in keys],'target_long')
