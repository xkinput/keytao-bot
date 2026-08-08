/** Promote one already-provisioned reserved rig account to local manager. */

const RESERVED_NAME_PREFIX = 'keytao-e2e-llm-rig-'
const RESERVED_EMAIL_SUFFIX = '@example.invalid'
const MIN_SYNTHETIC_QQ_DIGITS = 30

function requiredEnvironment(name: string): string {
  const value = process.env[name]?.trim() || ''
  if (!value) throw new Error(`Missing ${name}`)
  return value
}

function validateIdentity(name: string, email: string, qqId: string): void {
  if (!name.startsWith(RESERVED_NAME_PREFIX)) {
    throw new Error('Refusing to promote a non-reserved account name')
  }
  if (!email.endsWith(RESERVED_EMAIL_SUFFIX)) {
    throw new Error('Refusing to promote a non-reserved account email')
  }
  if (!/^\d+$/.test(qqId) || qqId.length < MIN_SYNTHETIC_QQ_DIGITS) {
    throw new Error('Refusing to promote a non-synthetic QQ binding')
  }
}

async function main(): Promise<void> {
  const name = requiredEnvironment('E2E_ADMIN_NAME')
  const email = requiredEnvironment('E2E_ADMIN_EMAIL')
  const qqId = requiredEnvironment('E2E_ADMIN_QQ_ID')
  validateIdentity(name, email, qqId)

  if (process.argv.includes('--validate-only')) {
    process.stdout.write('Reserved admin identity validation passed.\n')
    return
  }

  const { prisma } = await import('../../keytao-next/lib/prisma')
  try {
    const user = await prisma.user.findUnique({
      where: { name },
      select: {
        id: true,
        email: true,
        qqId: true,
        roles: { select: { id: true, value: true } },
      },
    })
    if (!user || user.email !== email || user.qqId !== qqId) {
      throw new Error('Reserved admin metadata does not match the provisioned account')
    }

    const roleValues = new Set(user.roles.map(role => role.value))
    if (!roleValues.has('R:NORMAL') || !roleValues.has('R:BOT')) {
      throw new Error('Reserved admin base account is missing dedicated bot roles')
    }

    const managerRole = await prisma.role.findUnique({
      where: { value: 'R:MANAGER' },
      select: { id: true },
    })
    if (!managerRole) {
      throw new Error('R:MANAGER is missing; run the keytao-next role initializer')
    }

    await prisma.user.update({
      where: { id: user.id },
      data: {
        status: 'ENABLE',
        roles: { connect: { id: managerRole.id } },
      },
    })
    process.stdout.write(`Reserved local manager provisioned: ${name}\n`)
  } finally {
    await prisma.$disconnect()
  }
}

main().catch(error => {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`)
  process.exitCode = 1
})
