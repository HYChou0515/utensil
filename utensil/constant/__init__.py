__all__ = ["PROJECT", "HOST_INFO", "LOG", 'TZ_TAIPEI', 'TZ_UTC']

import os
from configparser import ConfigParser, ExtendedInterpolation

from utensil.constant._constant import *
from utensil.general import open_utf8

if "UTENSIL_PROJECT_ROOT" in os.environ:
    PROJECT_ROOT = os.environ["UTENSIL_PROJECT_ROOT"]
else:
    PROJECT_ROOT = "."

if "UTENSIL_CONFIG" in os.environ:
    UTENSIL_CONFIG = os.environ["UTENSIL_CONFIG"]
else:
    UTENSIL_CONFIG = "utensil.ini"

config = ConfigParser(interpolation=ExtendedInterpolation())

config.read_string(f"""
[PROJECT]
ProjectRoot = {PROJECT_ROOT}
ConfigPath = {PROJECT_ROOT}/{UTENSIL_CONFIG}
ProjectName = .utensil
ProjectAbbr = ${{ProjectName}}
ProjectState = dev

[HOST_INFO]
HostName = localhost

[LOG]
Dir = {PROJECT_ROOT}/${{PROJECT:ProjectAbbr}}/log
Stream = info
Syslog = notset
File = info
FilePrefix = ${{Dir}}/${{PROJECT:ProjectAbbr}}.log
Level = info
MaxMessageLen = 60000
""")

if os.path.isfile(os.path.normpath(config["PROJECT"].get("ConfigPath"))):
    with open_utf8(os.path.normpath(config["PROJECT"].get("ConfigPath")),
                   'r') as f:
        config.read_file(f)

config["PROJECT"]["ProjectRoot"] = os.path.normpath(
    config["PROJECT"]["ProjectRoot"])
config["PROJECT"]["ProjectName"] = os.path.normpath(
    config["PROJECT"]["ProjectName"])
config["PROJECT"]["ProjectAbbr"] = os.path.normpath(
    config["PROJECT"]["ProjectAbbr"])
config["PROJECT"]["ConfigPath"] = os.path.normpath(
    config["PROJECT"]["ConfigPath"])

config["LOG"]["Dir"] = os.path.normpath(config["LOG"]["Dir"])
config["LOG"]["FilePrefix"] = os.path.normpath(config["LOG"]["FilePrefix"])

PROJECT = config["PROJECT"]
HOST_INFO = config["HOST_INFO"]
LOG = config["LOG"]

del os, ConfigParser, ExtendedInterpolation
del PROJECT_ROOT
del UTENSIL_CONFIG
