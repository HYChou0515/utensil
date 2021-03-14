import dataclasses
import logging
from dataclasses import dataclass

import utensil
from utensil import constant
from utensil.general.logger import parse_log_level

logger = utensil.get_logger(__name__)


DEFAULT_CONNECTOR_VERBOSE = logging.INFO


@dataclass
class ConnectorInformation:
    ip: str
    port: int
    user: str
    pds: str = dataclasses.field(repr=False)  # to protect password accidentally shown in log
    database: str


@dataclass
class ConnectorPoolInformation:
    connector_info: ConnectorInformation
    pool_size: int
    verbose: int


def get_connector_pool_info(tag_name: str):
    TAG_PREFIX = 'CONNECTION_INFO_OF_'
    trial_tags = []
    if tag_name.startswith(TAG_PREFIX):
        trial_tags.append(tag_name)
    trial_tags.append(TAG_PREFIX + tag_name)

    for trial in trial_tags:
        if constant.config.has_section(trial):
            connection_info_config = constant.config[trial]
            return ConnectorPoolInformation(
                connector_info=ConnectorInformation(
                    ip=connection_info_config.get('Ip'),
                    port=int(connection_info_config.get('Port')),
                    user=connection_info_config.get('User'),
                    pds=connection_info_config.get('Password'),
                    database=connection_info_config.get('ServiceName')
                ),
                pool_size=int(connection_info_config.get('PoolSize', 1)),
                verbose=parse_log_level(connection_info_config.get('Verbose', DEFAULT_CONNECTOR_VERBOSE))
            )
    err_str = '" and "'.join(trial_tags)
    raise ValueError(f'Cannot find connection info match "{err_str}".')
