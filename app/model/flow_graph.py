from typing import List, Tuple, Literal, Dict, Union, Any, Optional

from pydantic import Field

from model import Model

GraphObjectId = str


class MFlowGraphLink(Model):
    id: GraphObjectId
    source: GraphObjectId
    sourcePort: GraphObjectId
    target: GraphObjectId
    targetPort: GraphObjectId


class MFlowGraphPort(Model):
    id: GraphObjectId
    isIn: bool = Field(alias="in")
    name: str
    parentNode: GraphObjectId
    links: List[GraphObjectId]


class MFlowGraphNode(Model):
    id: GraphObjectId
    nodeType: Literal['task', 'switch-on', 'end-of-flow']
    module: Optional[str]
    params: List[Tuple[str, Literal['required', 'optional']]]
    ports: List[MFlowGraphPort]


class MFlowGraphModel(Model):
    models: Dict[GraphObjectId, Union[MFlowGraphLink, MFlowGraphNode]]


class MFlowGraph(Model):
    """
    >>> MFlowGraph.parse_obj(MFlowGraph.Config.schema_extra["example"])
    MFlowGraph(...)
    """
    layers: List[MFlowGraphModel]

    class Config(Model.Config):
        schema_extra = {
            "example": {
                'layers': [{
                    'models': {
                        'f7a24266-e53f-4efc-8ec0-1040fbe16da2': {
                            'id':
                                'f7a24266-e53f-4efc-8ec0-1040fbe16da2',
                            'source':
                                '8560a6b3-080b-483f-89b2-ed81724fbce7',
                            'sourcePort':
                                'e71e4fc1-886a-4af6-a1a1-296be042223f',
                            'target':
                                '8c93173f-de95-45d6-9dcb-e40e0897dc62',
                            'targetPort':
                                '8c7c0a1f-793f-487c-b99a-891e61f3e0b7'
                        },
                        '5c1a5131-5b43-4a1f-ac98-07404521532e': {
                            'id':
                                '5c1a5131-5b43-4a1f-ac98-07404521532e',
                            'source':
                                '8c93173f-de95-45d6-9dcb-e40e0897dc62',
                            'sourcePort':
                                '708637bd-ecea-4e51-8a00-d19772312eaa',
                            'target':
                                '369c73a7-2744-49cc-8ef2-3511de070e5f',
                            'targetPort':
                                '22d4c01f-4304-4f91-b7b1-9b611dde8045'
                        }
                    }
                }, {
                    'models': {
                        '8560a6b3-080b-483f-89b2-ed81724fbce7': {
                            'id':
                                '8560a6b3-080b-483f-89b2-ed81724fbce7',
                            'nodeType':
                                'switch-on',
                            'module':
                                None,
                            'params': [],
                            'ports': [{
                                'id':
                                    'e71e4fc1-886a-4af6-a1a1-296be042223f',
                                'in':
                                    False,
                                'name':
                                    'out',
                                'parentNode':
                                    '8560a6b3-080b-483f-89b2-ed81724fbce7',
                                'links': [
                                    'f7a24266-e53f-4efc-8ec0-1040fbe16da2'
                                ]
                            }]
                        },
                        '8c93173f-de95-45d6-9dcb-e40e0897dc62': {
                            'id':
                                '8c93173f-de95-45d6-9dcb-e40e0897dc62',
                            'nodeType':
                                'task',
                            'module':
                                'utensil.loopflow.functions.basic',
                            'params': [('default', 'required')],
                            'ports': [{
                                'id':
                                    '3c1558be-b7ff-4c25-b80a-1ff8a9f9fa68',
                                'in':
                                    True,
                                'name':
                                    'o',
                                'parentNode':
                                    '8c93173f-de95-45d6-9dcb-e40e0897dc62',
                                'links': []
                            }, {
                                'id':
                                    '8c7c0a1f-793f-487c-b99a-891e61f3e0b7',
                                'in':
                                    True,
                                'name':
                                    'trigger',
                                'parentNode':
                                    '8c93173f-de95-45d6-9dcb-e40e0897dc62',
                                'links': [
                                    'f7a24266-e53f-4efc-8ec0-1040fbe16da2'
                                ]
                            }, {
                                'id':
                                    '708637bd-ecea-4e51-8a00-d19772312eaa',
                                'in':
                                    False,
                                'name':
                                    'out',
                                'parentNode':
                                    '8c93173f-de95-45d6-9dcb-e40e0897dc62',
                                'links': [
                                    '5c1a5131-5b43-4a1f-ac98-07404521532e'
                                ]
                            }]
                        },
                        '369c73a7-2744-49cc-8ef2-3511de070e5f': {
                            'id':
                                '369c73a7-2744-49cc-8ef2-3511de070e5f',
                            'nodeType':
                                'end-of-flow',
                            'module':
                                None,
                            'params': [],
                            'ports': [{
                                'id':
                                    '22d4c01f-4304-4f91-b7b1-9b611dde8045',
                                'in':
                                    True,
                                'name':
                                    'trigger',
                                'parentNode':
                                    '369c73a7-2744-49cc-8ef2-3511de070e5f',
                                'links': [
                                    '5c1a5131-5b43-4a1f-ac98-07404521532e'
                                ]
                            }]
                        }
                    }
                }]
            }
        }
