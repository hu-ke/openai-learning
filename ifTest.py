# is_uploading = client.get('IS_UPLOADING')
# if is_uploading == 'YES':
#     return {
#         'code': '200',
#         'msg': 'server is busy try again later.',
#         'data': ''
#     }
# client.set('IS_UPLOADING', 'YES')
# result = 0
# for i in range(10**7):
#     result += i * i

# client.set('IS_UPLOADING', 'NO')
# return {
#     'code': '200',
#     'msg': 'success',
#     'data': result
# }