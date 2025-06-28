import random
from src.mutation.base.api_utils import APIFinder, APIReplacer, join_new_api
from src.mutation.base.api_mutator import *
from src.tools.enum_class import Framework
from src.tools.logger_utils import LoggerUtils
import src.tools.utils as utils


logger_instance = LoggerUtils.get_instance()
logger = logger_instance.logger

MAX_ATTEMPTS = 80


class TorchAPIMutator(APIMutator):

    def __init__(self,**kwargs):
        framework = kwargs.get("framework", Framework.PYTORCH)
        supported_apis = kwargs.get("supported_apis", None)
        super().__init__(framework=framework,supported_apis=supported_apis)
        self.relation_api_mapping = utils.read_pkl(os.path.join(root_path, "data", framework, "torch-relative_api_dict.pkl"))

    def api_replace_mutation(self, source_code, mutation_point_item, invoke_name2full_name):
        tree = ast.parse(source=source_code, mode='exec')
        api_call_dict = mutation_point_item["api_call_dict"]
        target_tuple = list(api_call_dict.items())[0]
        invoke_name = target_tuple[0] if target_tuple[1] == "" else ".".join(list(api_call_dict.items())[0])
      
        if invoke_name not in invoke_name2full_name.keys():
            return astor.to_source(tree)
        else:
            api_full_name = invoke_name2full_name[invoke_name]

        # replace the target api with the other api in the same namespace
        yaml_list = os.listdir(os.path.join(root_path, FUNC_SIG_PATH[self.framework]))
        api_root_path = api_full_name[:api_full_name.rfind('.')]
        similar_root_apis = []
        # NOTE: we try to extend the set of api replacement candidates.
        # If the api is torch, it means the api is like `torch.xxx`, we do not extend the similar_root_apis
        # since the apis under torch packages have very different parameters
        if api_root_path != "torch":
            for yaml_item in yaml_list:
                yaml_name = yaml_item.replace(".yaml", "")
                if yaml_name[:yaml_name.rfind('.')] == api_root_path and yaml_name != api_full_name:
                    similar_root_apis.append(yaml_name)

        signature_data = yaml.safe_load(
            utils.read_text(os.path.join(root_path, FUNC_SIG_PATH[self.framework], f"{api_full_name}.yaml")))
        constraint_data = yaml.safe_load(
            utils.read_text(
                os.path.join(root_path, DOCTER_PATH[self.framework], f"{api_full_name}.yaml".lower())))

        # View the position of the API in the ast of the test case And randomly select one for API replacement
        target = mutation_point_item["target"]
        target_position = mutation_point_item["point"]
        para_alignments, param_category = categorize_parameters(signature_data["input_params"])
        api_finder = APIFinder(target, invoke_name, api_full_name, signature_data, constraint_data,target_position,para_alignments, param_category)
        api_finder.visit(tree)
        old_param_dtype_dict = api_finder.param_dtype_dict
        logger.debug(f"param_dtype_dict: {old_param_dtype_dict}")

        if api_full_name in self.relation_api_mapping.keys() or len(similar_root_apis) != 0:
            # Select the corresponding API in DeepREL and filter for those with constraints
            if api_full_name in self.relation_api_mapping.keys():
                relation_apis = self.relation_api_mapping[api_full_name]
            else:
                relation_apis = []
            available_relation_apis = [relation_api for relation_api in relation_apis if
                                       relation_api in self.supported_apis]

            available_relation_apis.extend([similar_root_api for similar_root_api in similar_root_apis if
                                           similar_root_api in self.supported_apis])
            available_relation_apis = list(set(available_relation_apis))
            random.shuffle(available_relation_apis)
            logger.debug(f"api_full_name {api_full_name}")
            logger.debug(f"available_relation_apis {available_relation_apis}")

            if 'torch.Tensor' in api_full_name:
                available_relation_apis = [available_relation_api for available_relation_api in
                                           available_relation_apis if
                                           'torch.Tensor' in available_relation_api]
            else:
                available_relation_apis = [available_relation_api for available_relation_api in
                                           available_relation_apis if
                                           'torch.Tensor' not in available_relation_api]
            new_code_list  = []
            for relation_api in available_relation_apis:
                logger.debug(f"relation_api: {relation_api}")
                replace_signature_data = yaml.safe_load(
                    utils.read_text(os.path.join(root_path, FUNC_SIG_PATH[self.framework], f"{relation_api}.yaml")))
                replace_constraint_data = yaml.safe_load(
                    utils.read_text(
                        os.path.join(root_path, DOCTER_PATH[self.framework], f"{relation_api}.yaml".lower())))

                # Build a parameter dictionary based on the optional parameters of the API such as {'input': 'torch.tensor'}
                new_require_param_dtype_dict = {}
                for signature in replace_signature_data['input_params']:
                    if signature['require']:
                        param_name = signature['name']
                        dtype_name = replace_constraint_data['constraints'][param_name]['dtype']
                        # for the torch.Tensor, we do not need the input parameter
                        if param_name == "input" and "torch.Tensor" in relation_api:
                            continue
                        new_require_param_dtype_dict[param_name] = dtype_name
                logger.debug(f"old_param_dtype_dict: {old_param_dtype_dict}")
                logger.debug(f"new_require_param_dtype_dict: {new_require_param_dtype_dict}")

                if 'torch.Tensor' in relation_api:
                    relation_api = relation_api.replace('torch.Tensor', next(iter(target)))
                # Splice and replace the API, and then replace according to the ast
                new_replace_api, success = join_new_api(relation_api, old_param_dtype_dict,
                                                        new_require_param_dtype_dict,para_alignments)
                if not success:
                    logger.debug(f"{relation_api} conversion not supported")
                    continue
                logger.debug(f"new_replace_api: {new_replace_api}")

                api_replacer = APIReplacer(new_replace_api, target_position)
                new_code = api_replacer.conduct_mutation(source_code)
                new_code_list.append(new_code)
            return new_code_list

        else:
            # covert `y=func(x) to y=x`
            if len(old_param_dtype_dict.keys()) == 1:
                param_key = list(old_param_dtype_dict.keys())[0]
                if 'return' in signature_data.keys() and \
                        signature_data['return'] in old_param_dtype_dict[param_key]:
                    new_code = astor.to_source(tree)
                    new_code_replace = new_code.replace(f"{invoke_name}({param_key})", param_key)
                    return new_code_replace

        return astor.to_source(tree)






