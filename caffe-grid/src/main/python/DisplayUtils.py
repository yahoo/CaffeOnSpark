from PIL import Image
from io import BytesIO
from IPython.display import HTML
import numpy as np
from base64 import b64encode
from draw_net import parse_args

def get_np_array(row):
    return np.frombuffer(row.data, 'uint8').reshape((row.height,row.width))

def image_tag(np_array): 
    im = Image.fromarray(np_array, 'L')
    bytebuffer = BytesIO()
    im.save(bytebuffer, format='png')
    return "<img src='data:image/png;base64," + b64encode(bytebuffer.getvalue()) + "' />"

def show_df(df, nrows=10):
    data = df.take(nrows)
    html = "<table><tr><th>Index</th><th>Label</th><th>Image</th>"
    for i in range(nrows):
        row = data[i]
        html += "<tr>"
        html += "<td>%s</td>" % row.id
        html += "<td>%s</td>" % row.label
        html += "<td>%s</td>" % image_tag(get_np_array(row))
        html += "</tr>"
    html += "</table>"
    return HTML(html)

def show_network(input_net_proto_file, output_image_file):
    args = parse_args()
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(args.input_net_proto_file).read(), net)
    print('Drawing net to %s' % args.output_image_file)
    caffe.draw.draw_net_to_file(net, args.output_image_file, args.rankdir)
