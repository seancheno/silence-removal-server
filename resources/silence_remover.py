from flask import request
from flask_restful import Resource, reqparse

from models.silence_remover import SilenceRemoverModel

import os


class SilenceRemover(Resource):
    def get(self):
        return os.getcwd()

    def post(self):
        data = request.files['audioFile']
        return SilenceRemoverModel(data).run(), 200
