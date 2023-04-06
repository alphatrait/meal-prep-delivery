from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from flask_cors import CORS
import pandas as pd
from optimizer import optimizer
from config import auth

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
api = Api(app, version='1.0', title='RateItUp',
    description=''
)

ns = api.namespace('api/get-lat-lng', description='Address')
address = api.model('Address', {
    # 'id': fields.Integer(readonly=True, description='The task unique identifier'),
    'street': fields.String(required=True, description='The task details'),
    'zip': fields.String(required=True, description='The task details'),
    'city': fields.String(required=True, description='The task details'),
    'country': fields.String(required=True, description='The task details'),
    'order_id': fields.String(required=True, description='The task details')
    
})

address_list_fields = api.model('AddressList', {
    'data': fields.List(fields.Nested(address)),
})


@ns.route('/')
class UserManagement(Resource):
    @auth.login_required
    @ns.expect(address_list_fields)
    def post(self):
        '''Get the latitude and longitude'''
        df = pd.DataFrame(request.json["data"])
        optimized = optimizer(df)
        return optimized

if __name__ == '__main__':
    app.run(debug=True, port=5003)
